"""Adapted from https://github.com/SymbioticLab/Oort"""

import math
from collections import OrderedDict
from logging import DEBUG, INFO, WARNING, log
from random import Random
from typing import List

import numpy as np2


class OortSelector:
    """Oort's training selector
    """
    def __init__(self, sample_seed=233):
        self.totalArms = OrderedDict()
        self.training_round = 0

        self.exploration = 0.9  # args.exploration_factor
        self.decay_factor = 0.95  # args.exploration_decay
        self.exploration_min = 0.2  # args.exploration_min
        self.alpha = 0.3  # args.exploration_alpha

        self.rng = Random()
        self.rng.seed(sample_seed)
        self.unexplored = set()
        self.round_threshold = 10  # args.round_threshold
        self.round_prefer_duration = float('inf')
        self.last_util_record = 0

        # self.args = args
        self.pacer_step = 20
        self.pacer_delta = 5
        self.blacklist_rounds = -1
        self.blacklist_max_len = 0.3
        self.clip_bound = 0.98
        self.round_penalty = 2.0
        self.cut_off_util = 0.7

        self.sample_window = 5.  # args.sample_window
        self.exploitUtilHistory = []
        self.exploreUtilHistory = []
        self.exploitClients = []
        self.exploreClients = []
        self.successfulClients = set()
        self.blacklist = None

        np2.random.seed(sample_seed)

    def register_client(self, clientId, size, duration=1):
        # Initiate the score for arms. [score, time_stamp, # of trials, size of client, auxi, duration]
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {}
            self.totalArms[clientId]['reward'] = size
            self.totalArms[clientId]['duration'] = duration
            self.totalArms[clientId]['time_stamp'] = self.training_round
            self.totalArms[clientId]['count'] = 0
            self.totalArms[clientId]['status'] = True
            self.unexplored.add(clientId)

    def calculateSumUtil(self, clientList):
        cnt, cntUtil = 1e-4, 0
        for client in clientList:
            if client in self.successfulClients:
                cnt += 1
                cntUtil += self.totalArms[client]['reward']
        return cntUtil/cnt

    def pacer(self):
        # summarize utility in last epoch
        lastExplorationUtil = self.calculateSumUtil(self.exploreClients)
        lastExploitationUtil = self.calculateSumUtil(self.exploitClients)

        self.exploreUtilHistory.append(lastExplorationUtil)
        self.exploitUtilHistory.append(lastExploitationUtil)

        self.successfulClients = set()

        if self.training_round >= 2 * self.pacer_step and self.training_round % self.pacer_step == 0:
            utilLastPacerRounds = sum(self.exploitUtilHistory[-2*self.pacer_step:-self.pacer_step])
            utilCurrentPacerRounds = sum(self.exploitUtilHistory[-self.pacer_step:])

            # Cumulated statistical utility becomes flat, so we need a bump by relaxing the pacer
            if abs(utilCurrentPacerRounds - utilLastPacerRounds) <= utilLastPacerRounds * 0.1:
                self.round_threshold = min(100., self.round_threshold + self.pacer_delta)
                self.last_util_record = self.training_round - self.pacer_step
                log(DEBUG, "Training selector: Pacer changes at {} to {}".format(self.training_round, self.round_threshold))

            # change sharply -> we decrease the pacer step
            elif abs(utilCurrentPacerRounds - utilLastPacerRounds) >= utilLastPacerRounds * 5:
                self.round_threshold = max(self.pacer_delta, self.round_threshold - self.pacer_delta)
                self.last_util_record = self.training_round - self.pacer_step
                log(DEBUG, "Training selector: Pacer changes at {} to {}".format(self.training_round, self.round_threshold))

            log(DEBUG, "Training selector: utilLastPacerRounds {}, utilCurrentPacerRounds {} in round {}"
                .format(utilLastPacerRounds, utilCurrentPacerRounds, self.training_round))

        log(INFO, "Training selector: Pacer {}: lastExploitationUtil {}, lastExplorationUtil {}, last_util_record {}".
                        format(self.training_round, lastExploitationUtil, lastExplorationUtil, self.last_util_record))

    def update_client_util(self, clientId, reward, time_stamp=0, duration=1.):
        '''
        @ feedbacks['reward']: statistical utility
        @ feedbacks['duration']: system utility
        @ feedbacks['count']: times of involved
        '''
        self.totalArms[clientId]['reward'] = reward
        self.totalArms[clientId]['duration'] = duration
        self.totalArms[clientId]['time_stamp'] = time_stamp
        self.totalArms[clientId]['count'] += 1
        self.totalArms[clientId]['status'] = True

        self.unexplored.discard(clientId)
        self.successfulClients.add(clientId)

    def get_blacklist(self):
        blacklist = []
        if self.blacklist_rounds != -1:
            sorted_client_ids = sorted(list(self.totalArms), reverse=True, key=lambda k:self.totalArms[k]['count'])
            for clientId in sorted_client_ids:
                if self.totalArms[clientId]['count'] > self.blacklist_rounds:
                    blacklist.append(clientId)
                else:
                    break

            # we need to back up if we have blacklisted all clients
            predefined_max_len = self.blacklist_max_len * len(self.totalArms)

            if len(blacklist) > predefined_max_len:
                log(WARNING, "Training Selector: exceeds the blacklist threshold")
                blacklist = blacklist[:predefined_max_len]

        return set(blacklist)

    def select_participant(self, num_of_clients: int, feasible_clients: List[str]):
        if len([c for c in self.totalArms.values() if c['count'] > 0]) < num_of_clients:
            self.rng.shuffle(feasible_clients)
            client_len = min(num_of_clients, len(feasible_clients))
            clients = feasible_clients[:client_len]
        else:
            clients = self.getTopK(num_of_clients, feasible_clients)

        for item in clients:
            assert (item in feasible_clients)

        self.training_round += 1
        return clients

    def update_duration(self, clientId, duration):
        if clientId in self.totalArms:
            self.totalArms[clientId]['duration'] = duration

    def getTopK(self, numOfSamples, feasible_clients):
        self.blacklist = self.get_blacklist()

        self.pacer()

        # normalize the score of all arms: Avg + Confidence
        scores = {}
        numOfExploited = 0
        exploreLen = 0

        client_list = list(self.totalArms.keys())
        orderedKeys = [x for x in client_list if x in feasible_clients and x not in self.blacklist]

        if self.round_threshold < 100.:
            sortedDuration = sorted([self.totalArms[key]['duration'] for key in client_list])
            self.round_prefer_duration = sortedDuration[min(int(len(sortedDuration) * self.round_threshold/100.), len(sortedDuration)-1)]
        else:
            self.round_prefer_duration = float('inf')
        print(f"Oort's preferred round duration is now {self.round_prefer_duration:.1f} min")

        moving_reward, staleness, allloss = [], [], {}

        for clientId in orderedKeys:
            if self.totalArms[clientId]['reward'] > 0:
                creward = self.totalArms[clientId]['reward']
                moving_reward.append(creward)
                staleness.append(self.training_round - self.totalArms[clientId]['time_stamp'])

        max_reward, min_reward, range_reward, avg_reward, clip_value = self.get_norm(moving_reward, self.clip_bound)
        max_staleness, min_staleness, range_staleness, avg_staleness, _ = self.get_norm(staleness, thres=1)

        for key in orderedKeys:
            # we have played this arm before
            if self.totalArms[key]['count'] > 0:
                creward = min(self.totalArms[key]['reward'], clip_value)
                numOfExploited += 1

                sc = (creward - min_reward)/float(range_reward) \
                    + math.sqrt(0.1*math.log(self.training_round)/self.totalArms[key]['time_stamp']) # temporal uncertainty

                # sc = (creward - min_reward)/float(range_reward) \
                #     + self.alpha*((cur_time-self.totalArms[key]['time_stamp']) - min_staleness)/float(range_staleness)

                clientDuration = self.totalArms[key]['duration']
                if clientDuration > self.round_prefer_duration:
                    sc *= ((float(self.round_prefer_duration)/max(1e-4, clientDuration)) ** self.round_penalty)

                if self.totalArms[key]['time_stamp'] == self.training_round:
                    allloss[key] = sc

                scores[key] = abs(sc)

        self.exploration = max(self.exploration*self.decay_factor, self.exploration_min)

        explorationLen = int(numOfSamples*self.exploration)

        # exploration
        unexplored = [x for x in list(self.unexplored) if x in feasible_clients]
        if len(unexplored) > 0:
            init_reward = {}
            for cl in unexplored:
                init_reward[cl] = self.totalArms[cl]['reward']
                clientDuration = self.totalArms[cl]['duration']

                if clientDuration > self.round_prefer_duration:
                    init_reward[cl] *= ((float(self.round_prefer_duration)/max(1e-4, clientDuration)) ** self.round_penalty)

            # prioritize w/ some rewards (i.e., size)
            exploreLen = min(len(unexplored), explorationLen)
            pickedUnexploredClients = sorted(init_reward, key=init_reward.get, reverse=True)[:min(int(self.sample_window*exploreLen), len(init_reward))]

            unexploredSc = float(sum([init_reward[key] for key in pickedUnexploredClients]))

            pickedUnexplored = list(np2.random.choice(pickedUnexploredClients, exploreLen,
                            p=[init_reward[key]/unexploredSc for key in pickedUnexploredClients], replace=False))

            self.exploreClients = pickedUnexplored
        else:
            self.exploreClients = []

        # exploitation
        clientLakes = list(scores.keys())
        exploitLen = min(numOfSamples-len(self.exploreClients), len(clientLakes))

        # take the top-k, and then sample by probability, take 95% of the cut-off loss
        sortedClientUtil = sorted(scores, key=scores.get, reverse=True)

        # take cut-off utility
        cut_off_util = scores[sortedClientUtil[-1]] * self.cut_off_util

        tempPickedClients = []
        for clientId in sortedClientUtil:
            # TODO this rule makes little sense if there are only few clients in total
            # we want at least 10 times of clients for augmentation
            if scores[clientId] < cut_off_util and len(tempPickedClients) > 10.*exploitLen:
                print(f"CUTOFF {clientId}")
                break
            tempPickedClients.append(clientId)

        augment_factor = len(tempPickedClients)

        # totalSc = max(1e-4, float(sum([scores[key] for key in tempPickedClients])))
        totalSc = float(sum([scores[key] for key in tempPickedClients]))
        self.exploitClients = list(np2.random.choice(tempPickedClients, exploitLen, p=[scores[key]/totalSc for key in tempPickedClients], replace=False))

        print(f"Exploration  ({len(self.exploreClients)}): {', '.join(self.exploreClients)}")
        print(f"Exploitation ({len(self.exploitClients)}): {', '.join(self.exploitClients)}")
        pickedClients = self.exploreClients + self.exploitClients

        top_k_score = []
        for clientId in pickedClients:
            _score = (self.totalArms[clientId]['reward'] - min_reward)/range_reward
            _staleness = self.alpha*((self.training_round-self.totalArms[clientId]['time_stamp']) - min_staleness)/float(range_staleness) #math.sqrt(0.1*math.log(cur_time)/max(1e-4, self.totalArms[clientId]['time_stamp']))
            top_k_score.append((self.totalArms[clientId], [_score, _staleness]))

        log(INFO, "At round {}, UCB exploited {}, augment_factor {}, exploreLen {}, un-explored {}, exploration {}, round_threshold {}, sampled score is {}"
            .format(self.training_round, numOfExploited, augment_factor/max(1e-4, exploitLen), exploreLen, len(self.unexplored), self.exploration, self.round_threshold, top_k_score))
        # logging.info("At time {}, all rewards are {}".format(cur_time, allloss))

        return pickedClients

    def get_norm(self, aList, clip_bound=0.95, thres=1e-4):
        aList.sort()
        clip_value = aList[min(int(len(aList)*clip_bound), len(aList)-1)]

        _max = aList[-1]
        _min = aList[0]*0.999
        _range = max(_max - _min, thres)
        _avg = sum(aList)/max(1e-4, float(len(aList)))

        return float(_max), float(_min), float(_range), float(_avg), float(clip_value)
