package org.chrisvenator.Voting;

import org.chrisvenator.KNNClassifier;

import java.util.List;

public class WightedMajorityVoting implements VotingMetric {
    @Override
    public int vote(int k, List<KNNClassifier.Neighbor> neighbors) {
        return 0;
    }
}
