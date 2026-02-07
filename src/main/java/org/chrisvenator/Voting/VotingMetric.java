package org.chrisvenator.Voting;

import org.chrisvenator.KNNClassifier;

import java.util.List;

public interface VotingMetric {
    int vote(int k, List<KNNClassifier.Neighbor> neighbors);
}
