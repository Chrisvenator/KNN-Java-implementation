package org.chrisvenator.Voting;

import org.chrisvenator.KNNClassifier;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Weighted majority voting where closer neighbors have more influence.
 *
 * <p>Each neighbor's vote is weighted by the inverse of its distance.
 * Closer neighbors (smaller distance) get higher weights.</p>
 *
 * <p>Weight formula: weight = 1 / (distance + epsilon)
 * where epsilon prevents division by zero for identical points.</p>
 */
public class WeightedMajorityVoting implements VotingMetric {
    
    private static final double EPSILON = 1e-10;
    
    @Override
    public int vote(int k, List<KNNClassifier.Neighbor> neighbors) {
        if (neighbors == null || neighbors.isEmpty()) {
            throw new IllegalArgumentException("Neighbors list cannot be null or empty");
        }
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        if (k > neighbors.size()) {
            throw new IllegalArgumentException(
                    String.format("k (%d) cannot exceed number of neighbors (%d)", k, neighbors.size())
            );
        }
        
        // Calculate weighted votes for each label
        Map<Integer, Double> weightedVotes = new HashMap<>();
        
        for (int i = 0; i < k; i++) {
            KNNClassifier.Neighbor neighbor = neighbors.get(i);
            int label = neighbor.getLabel();
            double distance = neighbor.getDistance();
            
            // Weight = 1 / (distance + epsilon)
            // Closer neighbors have higher weight
            double weight = 1.0 / (distance + EPSILON);
            
            weightedVotes.merge(label, weight, Double::sum);
        }
        
        // Return label with highest weighted vote
        return weightedVotes.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElseThrow(() -> new IllegalStateException("Failed to determine winner"));
    }
}