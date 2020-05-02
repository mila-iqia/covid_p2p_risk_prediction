import datetime
from collections import defaultdict

from covid19sim.frozen.utils import Message, decode_message, encode_message,\
    decode_update_message, hash_to_cluster, hash_to_cluster_day


class Clusters:
    """ This class manages the storage and clustering of messages and message updates.
     You can think of each message as database record, with the message updates indicating
     an update to a record. We have various clues (uid, risk, number of updates sent for a day) as
     signals about which records to update, as well as whether our clusters are correct."""

    def __init__(self):
        self.num_messages = 0
        self.clusters = defaultdict(list)
        self.clusters_by_day = defaultdict(dict)

    def add_messages(self, messages, current_day, rng=None):
        """ This function clusters new messages by scoring them against old messages in a sort of naive nearest neighbors approach"""
        for message in messages:
            m_dec = decode_message(message)
            best_cluster, _, best_score = self.score_matches(m_dec, current_day, rng=rng)
            self.num_messages += 1
            self.clusters[best_cluster].append(message)
            self.add_to_clusters_by_day(best_cluster, m_dec.day, message)

    def score_matches(self, m_new, current_day, rng=None):
        """ This function checks a new risk message against all previous messages, and assigns to the closest one in a brute force manner"""
        best_score = 2
        cluster_days = hash_to_cluster_day(m_new)
        best_cluster = hash_to_cluster(m_new)

        if self.clusters_by_day[current_day].get(best_cluster, None):
            return (best_cluster, None, 3)
        found = False
        for day, cluster_ids in cluster_days.items():
            for cluster_id in cluster_ids:
                if self.clusters_by_day[current_day - day].get(cluster_id, None):
                    best_cluster = cluster_id
                    found = True
                    break
            if found:
                break
            best_score -= 1

        return best_cluster, None, best_score

    def update_records(self, update_messages):
        # if we're using naive tracing, we actually don't care which records we update
        if not update_messages:
            return self

        grouped_update_messages = self.group_by_received_at(update_messages)
        for received_at, update_messages in grouped_update_messages.items():
            old_cluster = None
            for update_message in update_messages:
                old_message_dec = Message(update_message.uid, update_message.risk, update_message.day, update_message.unobs_id, update_message.has_app)
                old_message_enc = encode_message(old_message_dec)
                updated_message = Message(old_message_dec.uid, update_message.new_risk, old_message_dec.day, old_message_dec.unobs_id, old_message_dec.has_app)
                new_cluster = hash_to_cluster(updated_message)
                self.update_record(old_cluster, new_cluster, old_message_dec, updated_message)
        return self

    def update_record(self, old_cluster_id, new_cluster_id, message, updated_message):
        """ This function updates a message in all of the data structures and can change the cluster that this message is in"""
        old_m_enc = encode_message(message)
        new_m_enc = encode_message(updated_message)

        del self.clusters[old_cluster_id][self.clusters[old_cluster_id].index(old_m_enc)]
        del self.clusters_by_day[message.day][old_cluster_id][
            self.clusters_by_day[message.day][old_cluster_id].index(old_m_enc)]

        self.clusters[new_cluster_id].append(encode_message(updated_message))
        self.add_to_clusters_by_day(new_cluster_id, updated_message.day, new_m_enc)

    def add_to_clusters_by_day(self, cluster, day, m_i_enc):
        if self.clusters_by_day[day].get(cluster):
            self.clusters_by_day[day][cluster].append(m_i_enc)
        else:
            self.clusters_by_day[day][cluster] = [m_i_enc]

    def group_by_received_at(self, update_messages):
        """ This function takes in a set of update messages received during some time interval and clusters them based on how near in time they were received"""
        # TODO: We need more information about the actual implementation of the message protocol to use this.\
        # TODO: it is possible that received_at is actually the same for all update messages under the protocol, in which case we can delete this function.
        TIME_THRESHOLD = datetime.timedelta(minutes=1)
        grouped_messages = defaultdict(list)
        for m1 in update_messages:
            m1 = decode_update_message(m1)
            # if m1.received_at - received_at < TIME_THRESHOLD and -(m1.received_at - received_at) < TIME_THRESHOLD:
            #     grouped_messages[received_at].append(m1)
            # else:
            grouped_messages[m1.received_at].append(m1)

        return grouped_messages

    def purge(self, current_day):
        for cluster_id, messages in self.clusters_by_day[current_day - 14].items():
            for message in messages:
                del self.clusters[cluster_id][self.clusters[cluster_id].index(message)]
                self.num_messages -= 1
        to_purge = []
        for cluster_id, messages in self.clusters.items():
            if len(self.clusters[cluster_id]) == 0:
                to_purge.append(cluster_id)
        for cluster_id in to_purge:
            del self.clusters[cluster_id]
        if current_day - 14 >= 0:
            del self.clusters_by_day[current_day - 14]
        to_purge = defaultdict(list)
        for day, clusters in self.clusters_by_day.items():
            for cluster_id, messages in clusters.items():
                if not messages:
                    to_purge[day].append(cluster_id)
        for day, cluster_ids in to_purge.items():
            for cluster_id in cluster_ids:
                del self.clusters_by_day[day][cluster_id]
        self.update_messages = []

    def __len__(self):
        return self.num_messages
