package org.chrisvenator.Voting;

import org.chrisvenator.KNNClassifier;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class MajorityVoting implements VotingMetric {
    @Override
    public int vote(int k, List<KNNClassifier.Neighbor> neighbors) {
        return neighbors.subList(0, k).stream()
                .collect(Collectors.groupingBy(KNNClassifier.Neighbor::getLabel, Collectors.counting()))
                .entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElseThrow(() -> new IllegalArgumentException("No neighbors to vote"));
    }
}

