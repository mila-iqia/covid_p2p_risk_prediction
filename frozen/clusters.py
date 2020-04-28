import datetime
import numpy as np
import operator
from collections import defaultdict
from scipy.stats import wasserstein_distance as dist
from frozen.utils import Message, decode_message, encode_message, decode_update_message, hash_to_cluster, compare_uids


class Clusters:
    """ This class manages the storage and clustering of messages and message updates.
     You can think of each message as database record, with the message updates indicating
     an update to a record. We have various clues (uid, risk, number of updates sent for a day) as
     signals about which records to update, as well as whether our clusters are correct."""

    def __init__(self):
        self.all_messages = []
        self.clusters = defaultdict(list)
        self.clusters_by_day = defaultdict(dict)

    def add_to_clusters_by_day(self, cluster, day, m_i_enc):
        if self.clusters_by_day[day].get(cluster):
            self.clusters_by_day[day][cluster].append(m_i_enc)
        else:
            self.clusters_by_day[day][cluster] = [m_i_enc]

    def add_messages(self, messages, current_day, rng=None):
        """ This function clusters new messages by scoring them against old messages in a sort of naive nearest neighbors approach"""
        for message in messages:
            m_dec = decode_message(message)
            # otherwise score against previous messages
            best_cluster, best_message, best_score = self.score_matches(m_dec, current_day, rng=rng)
            if best_score >= 0:
                cluster_id = best_cluster
            else:
                cluster_id = hash_to_cluster(m_dec)

            self.all_messages.append(message)
            self.clusters[cluster_id].append(message)
            self.add_to_clusters_by_day(cluster_id, m_dec.day, message)

    def score_matches(self, m_new, current_day, rng=None):
        """ This function checks a new risk message against all previous messages, and assigns to the closest one in a brute force manner"""
        best_cluster = hash_to_cluster(m_new)
        best_message = None
        best_score = -1
        for i in range(current_day - 3, current_day + 1, -1):
            for cluster_id, messages in self.clusters_by_day[i].items():
                for m_enc in messages:
                    obs_uid, risk, day, unobs_uid, has_app = decode_message(m_enc)
                    if m_new.uid == obs_uid and m_new.day == day:
                        best_cluster = cluster_id
                        best_message = m_enc
                        best_score = 3
                        break
                    elif compare_uids(m_new.uid, obs_uid, 1) and m_new.day - 1 == day and m_new.risk == risk:
                        best_cluster = cluster_id
                        best_message = m_enc
                        best_score = 2
                    elif compare_uids(m_new.uid, obs_uid, 2) and m_new.day - 2 == day and best_score < 1:
                        best_cluster = cluster_id
                        best_message = m_enc
                        best_score = 1
                    elif compare_uids(m_new.uid, obs_uid, 3) and m_new.day - 3 == day and best_score < 0:
                        best_cluster = cluster_id
                        best_message = m_enc
                        best_score = 0
                    else:
                        best_cluster = cluster_id
                        best_message = m_enc
                        best_score = -1
                if best_score == 3:
                    break
            if best_score == 3:
                break
        # print(f"best_cluster: {best_cluster}, m_new: {m_new}, best_score: {best_score}")
        # print(self.clusters)

        if best_message:
            best_message = decode_message(best_message)
        return best_cluster, best_message, best_score

    def score_matches_in_cluster(self, update_message, cluster_messages):
        """ This function takes in a set of messages and returns the best scoring one"""
        best_scores = []
        for m in cluster_messages:
            best_scores.append(self.score_two_messages(update_message, m))
        return max(best_scores)

    def score_two_messages(self, update_message, risk_message):
        """ This function takes in two messages and scores how well they match"""
        obs_uid, risk, day, unobs_uid = decode_message(risk_message)
        if update_message.uid == obs_uid and update_message.day == day and update_message.risk == risk:
            score = 3
        elif compare_uids(update_message.uid, obs_uid,
                          1) and update_message.day - 1 == day and update_message.risk == risk:
            score = 2
        elif compare_uids(update_message.uid, obs_uid,
                          2) and update_message.day - 2 == day and update_message.risk == risk:
            score = 1
        elif compare_uids(update_message.uid, obs_uid,
                          3) and update_message.day - 3 == day and update_message.risk == risk:
            score = 0
        else:
            score = -1
        return score

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

    def update_record(self, old_cluster_id, new_cluster_id, message, updated_message):
        """ This function updates a message in all of the data structures and can change the cluster that this message is in"""
        old_m_enc = encode_message(message)
        new_m_enc = encode_message(updated_message)
        del self.clusters[old_cluster_id][self.clusters[old_cluster_id].index(old_m_enc)]
        del self.all_messages[self.all_messages.index(old_m_enc)]
        del self.clusters_by_day[message.day][old_cluster_id][
            self.clusters_by_day[message.day][old_cluster_id].index(old_m_enc)]

        self.clusters[new_cluster_id].append(encode_message(updated_message))
        self.all_messages.append(new_m_enc)
        self.add_to_clusters_by_day(new_cluster_id, updated_message.day, new_m_enc)

    def update_records(self, update_messages, human):
        # if we're using naive tracing, we actually don't care which records we update
        if not update_messages:
            return self
        grouped_update_messages = self.group_by_received_at(update_messages)
        for received_at, update_messages in grouped_update_messages.items():

            for update_message in update_messages:
                best_cluster = hash_to_cluster(update_message)
                best_message = self.clusters_by_day[update_message.day][best_cluster][0]
                best_message = decode_message(best_message)
                updated_message = Message(best_message.uid, update_message.new_risk, best_message.day, best_message.unobs_id)
                self.update_record(best_cluster, best_cluster, best_message, updated_message)
        return self

    def purge(self, current_day):
        for cluster_id, messages in self.clusters_by_day[current_day - 14].items():
            for message in messages:
                del self.clusters[cluster_id][self.clusters[cluster_id].index(message)]
                del self.all_messages[self.all_messages.index(message)]
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
        return len(self.clusters.keys())

    @property
    def num_messages(self):
        return len(self.all_messages)
