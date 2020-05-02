# COVID-19 Simulator Inference Server

This is a fork off the Contact Tracing Transformer (CTT) inference code used to provide a
parallel solution for the processing of risk level estimation requests from the simulator.

## Links to other parts of this risk-management project

* [Contact Tracing Transformer](https://github.com/nasimrahaman/ctt): Baseline risk/infectiousness level prediction model.
* [Simulator](https://github.com/pg2455/covid_p2p_simulation): Generates the underlying data, using an epidemiolgically-informed agent-based model written in Simpy 
* [Data parser](TODO): Takes the logs from the Simulator and generates the supervised learning dataset, masking/discretizing some values in accordance with privacy concerns.
* [Transformer model](https://github.com/nasimrahaman/ctt): Full version of the model the example code in here is based on
* [GraphNN model](TODO): Implementation of a graph neural network for joint risk prediction and simulator optimizatio 
* [coVAEd model](TODO): Variational auto-encoder approach for joint risk prediction and model optimization
* [Decentralized Loopy Belief](TODO): Training of a graphical model of disease propagation via message-passing
