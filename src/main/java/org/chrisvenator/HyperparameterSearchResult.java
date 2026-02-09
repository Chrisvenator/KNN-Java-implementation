package org.chrisvenator;

import org.chrisvenator.Voting.VotingMetric;
import org.chrisvenator.distance.DistanceMetric;

import java.util.Map;

public record HyperparameterSearchResult(
        int bestK,
        DistanceMetric bestDistanceMetric,
        VotingMetric bestVotingMetric,
        double bestAccuracy,
        Map<String, Map<Integer, CrossValidationResult>> allResults
) {
    @Override
    public String toString() {
        return String.format("Best Hyperparameters:%n" +
                        "  k = %d%n" +
                        "  Distance Metric = %s%n" +
                        "  Voting Metric = %s%n" +
                        "  Accuracy = %.4f",
                bestK,
                bestDistanceMetric.toString(),
                bestVotingMetric.getClass().getSimpleName(),
                bestAccuracy);
    }
}