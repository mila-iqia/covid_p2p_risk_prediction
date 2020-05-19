"""
Script to convert a saved pytorch model (found at ./models/model.pth) to
a cdollection of TFLite models.

Each TFLite model expects a fixed number of messages. To obtain the predictions
for a new example, one should choose the TFLite model which will require the
least amount of padding (for computational and RAM reasons).

There is no direct way to convert from Pytorch to TFLite so this script
performs the following steps :
- Pytorch to ONNX
- ONNX to TF Graph
- TF Graph to TF Saved Model
- TF Saved Model to TFLite model

2020-02-27 NOTE : In the future, it may be possible to have a single TFLite
model which supports dynamic sizes but the current stable release of
Tensorflow (2.1) does not support this and the current release candidate (rc3)
for Tensorflow 2.2 doesn't appear to change this.

2020-02-27 WARNING : As of this writing, data format has yes to stabilize. If
the script gives shape errors, the arguments to the dataloader may need to be
changed to reflect what the saved pytorch model expects.
"""

import numpy
import os
import shutil

import onnx
from onnx_tf.backend import prepare
import torch
import torch.onnx
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from ctt.models.transformer import ContactTracingTransformer
from ctt.data_loading.loader import get_dataloader

NB_EXAMPLES_FOR_SANITY_CHECK = 10
NB_MESSAGES_BUCKETS = [100, 300, 1000, 3000, 10000]

def pad_pytorch_tensor(arr, axis, nb_zeroes):
    # Pads pytorch tensor with trailing zeroes across a single dimension
    pad_dimensions = [(0, nb_zeroes) if i==axis else (0, 0) for i in range(arr.ndim)]
    padded_arr = numpy.pad(arr, pad_dimensions, 'constant', constant_values=0)
    padded_tensor = torch.Tensor(padded_arr)
    return padded_tensor


def pad_messages_minibatch(batch, target_nb_messages):
    nb_messages = batch['mask'].shape[1]
    nb_messages_add = target_nb_messages - nb_messages

    # Initialize padded_batch as copy of batch
    padded_batch = batch.copy()

    # Pad the inputs whose size depends on the nb of messages
    padded_batch['mask'] = pad_pytorch_tensor(
        batch['mask'], axis=1, nb_zeroes=nb_messages_add)
    padded_batch['encounter_health'] = pad_pytorch_tensor(
        batch['encounter_health'], axis=1, nb_zeroes=nb_messages_add)
    padded_batch['encounter_message'] = pad_pytorch_tensor(
        batch['encounter_message'], axis=1, nb_zeroes=nb_messages_add)
    padded_batch['encounter_partner_id'] = pad_pytorch_tensor(
        batch['encounter_partner_id'], axis=1, nb_zeroes=nb_messages_add)
    padded_batch['encounter_day'] = pad_pytorch_tensor(
        batch['encounter_day'], axis=1, nb_zeroes=nb_messages_add)
    padded_batch['encounter_duration'] = pad_pytorch_tensor(
        batch['encounter_duration'], axis=1, nb_zeroes=nb_messages_add)
    padded_batch['encounter_is_contagion'] = pad_pytorch_tensor(
        batch['encounter_is_contagion'], axis=1, nb_zeroes=nb_messages_add)
    return padded_batch


def convert_pytorch_model(pytorch_model, working_directory="./tmp_tfmodel_conversion/",
                          dataset_path="./data/sim_v2_people-1000_days-30_init-0.003_seed-0_20200509-182246-output.zip"):

    for nb_messages in NB_MESSAGES_BUCKETS:
        convert_pytorch_model_fixed_messages(pytorch_model, nb_messages,
                                             working_directory, dataset_path)


def convert_pytorch_model_fixed_messages(pytorch_model, nb_messages,
                                         working_directory, dataset_path):

    # Make sure we are converting the inference graph
    pytorch_model.eval()

    # Setup working directory
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)

    # Load dataset (used for sanity checking the converted models)
    dataloader = get_dataloader(batch_size=1, shuffle=False, num_workers=0,
                                path=dataset_path)

    # Find a minibatch with, at most, the number of messages that we want to
    # create a TFLite model for. If it has less than that number of messages,
    # pad it so it has the right number.
    batch = None
    for b in iter(dataloader):
        if b['mask'].shape[1] <= nb_messages:
            batch = pad_messages_minibatch(b, nb_messages)
            break

    if batch is None:
        raise ValueError("Attempting to create TFLite model for a nb of "
                         "messages lower than anything in the dataset")

    # Get list of inputs names as in the batch
    input_names=[]
    for i in batch:
        input_names.append(i)
    output_names = ['encounter_variables', 'latent_variable']

    # Convert PyTorch model to ONNX format
    onnx_model_path = os.path.join(working_directory, "model_onnx_10.onnx")
    torch.onnx.export(pytorch_model,
                      batch,
                      onnx_model_path,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names)

    # Load ONNX model and convert to TF model
    onnx_model = onnx.load(onnx_model_path)
    tf_model = prepare(onnx_model)


    # Convert the tf graph to a TF Saved Model
    tf_model_path = os.path.join(working_directory, "tf_model" )
    if os.path.isdir(tf_model_path):
        print('Already saved a TF model, cleaning up')
        shutil.rmtree(tf_model_path, ignore_errors=True)

    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(tf_model_path)
    with tf.compat.v1.Session(graph=tf_model.graph) as sess:

        input_spec = {}
        output_spec = {}
        for name in tf_model.inputs:
            input_spec[name] = tf_model.tensor_dict[name]
        for name in output_names:
            output_spec[name] = tf_model.tensor_dict[name]

        sigs = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY :
                tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(input_spec, output_spec)}

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)
        builder.save()

    # Convert Saved Model to TFLite model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    #converter.allow_custom_ops=True
    converter.experimental_new_converter=True
    #converter.enable_mlir_converter=True
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # 8-bits weight quantization
    tflite_model = converter.convert()
    tflite_model_path = os.path.join(working_directory, "model_%i_messages.tflite" % nb_messages)
    open(tflite_model_path, "wb").write(tflite_model)

    # Sanity-check the Tensorflow and TFLite models on the examples that have, at most,
    # the maximum number of messages that they can handle.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    max_pytorch_tf_delta = -1
    max_tf_tflite_delta = -1
    max_pytorch_tflite_delta = -1
    nb_validation_samples = 0
    for batch in iter(dataloader):

        # If this example is too big for the model, skip it.
        batch_nb_messages = batch["mask"].shape[1]
        if batch_nb_messages > nb_messages:
            continue

        # Generate padded inputs for the TF and TFLite models
        padded_batch = pad_messages_minibatch(batch, nb_messages)

        # Get pytorch model output
        pytorch_output = pytorch_model(batch)
        pytorch_padded_output = pytorch_model(padded_batch)

        # Get TF model output
        tf_padded_output = tf_model.run(padded_batch)
        tf_output={
            "encounter_variables" : tf_padded_output.encounter_variables[:, :batch_nb_messages],
            "latent_variable" : tf_padded_output.latent_variable
        }

        # Send inputs to the TFLite model
        interpreter.allocate_tensors()
        for inp_detail in interpreter.get_input_details():
            inp_name = inp_detail["name"]
            interpreter.set_tensor(inp_detail["index"], padded_batch[inp_name].numpy())

        # Get TFLite model outputs
        tflite_output = {}
        interpreter.invoke()
        for out_name, out_detail in zip(output_names, interpreter.get_output_details()):
            if out_name == "encounter_variables":
                # Remove the message padding
                tflite_output[out_name] = interpreter.get_tensor(out_detail["index"])[:, :batch_nb_messages]
            else:
                tflite_output[out_name] = interpreter.get_tensor(out_detail["index"])

        # Compare the three models
        for k in pytorch_output.keys():
            k_pytorch_tf_delta = pytorch_output[k].detach().numpy() - tf_output[k]
            max_pytorch_tf_delta = max(max_pytorch_tf_delta,
                                       numpy.abs(k_pytorch_tf_delta).max())

            k_tf_tflite_delta = tf_output[k] - tflite_output[k]
            max_tf_tflite_delta = max(max_tf_tflite_delta,
                                      numpy.abs(k_tf_tflite_delta).max())

            k_pytorch_tflite_delta = pytorch_output[k].detach().numpy() - tflite_output[k]
            max_pytorch_tflite_delta = max(max_pytorch_tflite_delta,
                                           numpy.abs(k_pytorch_tflite_delta).max())

        # Limit the testing to avoid spending too much time on it.
        nb_validation_samples += 1
        if nb_validation_samples >= NB_EXAMPLES_FOR_SANITY_CHECK:
            break


    # Log the results of the conversion sanity check
    if nb_validation_samples == 0:
        raise ValueError("Model generated for %i messages could not be tested. All validation "
        "samples have too many messages to be used for this model." % nb_messages)

    log_filename = os.path.join(working_directory, "model_%i_messages.txt" % nb_messages)
    with open(log_filename, "w") as f:

        f.write("Models compared on %i validation samples\n\n" % nb_validation_samples)

        f.write("Conversion from pytorch model to TF model\n")
        f.write("  Max abs diff between outputs : %f \n\n" % max_pytorch_tf_delta)

        f.write("Conversion from TF model to TFLite model\n")
        f.write("  Max abs diff between outputs : %f \n\n" % max_tf_tflite_delta)

        f.write("Overall conversion from pytorch model to TFLite model\n")
        f.write("  Max abs diff between outputs : %f \n\n" % max_pytorch_tflite_delta)

    return max_pytorch_tflite_delta


if __name__ == '__main__':
    # Load pytorch model
    pytorch_model = ContactTracingTransformer()
    pytorch_model.load_state_dict(torch.load("models/model.pth"))
    pytorch_model.eval()

    # Launch conversion
    convert_pytorch_model(pytorch_model)