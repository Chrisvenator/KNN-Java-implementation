package org.chrisvenator;

import lombok.Getter;
import org.chrisvenator.Voting.MajorityVoting;
import org.chrisvenator.Voting.VotingMetric;
import org.chrisvenator.distance.DistanceMetric;
import org.chrisvenator.distance.EuclideanDistance;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * K-Nearest Neighbors (KNN) classifier implementation.
 *
 * <p>This classifier finds the k nearest neighbors to a test point and
 * predicts the label based on a voting strategy among those neighbors.</p>
 *
 * <p>Usage example:</p>
 * <pre>
 * KNNClassifier knn = new KNNClassifier();
 * knn.fit(trainingData, labels);
 * int prediction = knn.predict(5, testPoint, new EuclideanDistance(), new MajorityVoting());
 * </pre>
 */
public class KNNClassifier {
    private DataPoint[] trainingData;
    
    /**
     * Creates a new KNN classifier.<br>
     * Call fit(...) next
     */
    public KNNClassifier() {}
    
    /**
     * Trains the classifier with labeled data.
     *
     * @param trainingData 2D array where each row is a sample, each column is a feature
     * @param labels       Array of integer labels for each training sample
     * @throws IllegalArgumentException if inputs are null, lengths don't match,
     *                                  or if training data is empty
     */
    public void fit(double[][] trainingData, int[] labels) {
        validateFitInputs(trainingData, labels);
        
        this.trainingData = new DataPoint[trainingData.length];
        for (int i = 0; i < trainingData.length; i++) {
            this.trainingData[i] = new DataPoint(trainingData[i], labels[i]);
        }
    }
    
    /**
     * Predicts the label for a test point using k-nearest neighbors.
     *
     * @param k              Number of neighbors to consider (must be positive and ≤ training size)
     * @param testPoint      Point to classify
     * @param distanceMetric Distance metric to use (null defaults to Euclidean)
     * @param votingMetric   Voting strategy to use (null defaults to majority voting)
     * @return Predicted label
     * @throws IllegalStateException    if classifier hasn't been trained (fit not called)
     * @throws IllegalArgumentException if k is invalid or testPoint is incompatible
     */
    public int predict(int k, double[] testPoint, DistanceMetric distanceMetric, VotingMetric votingMetric) {
        validatePredictInputs(k, testPoint);
        
        // Use defaults if not provided
        if (distanceMetric == null) distanceMetric = new EuclideanDistance();
        if (votingMetric == null) votingMetric = new MajorityVoting();
        
        // Calculate distances to all training points
        List<Neighbor> neighbors = new ArrayList<>(trainingData.length);
        for (DataPoint dataPoint : trainingData) {
            double distance = distanceMetric.calculateDistance(dataPoint.getVector(), testPoint);
            neighbors.add(new Neighbor(dataPoint, distance));
        }
        
        // Sort by distance
        neighbors.sort(Comparator.comparingDouble(Neighbor::getDistance));
        
        // Vote using k nearest neighbors
        return votingMetric.vote(k, neighbors);
    }
    
    private void validateFitInputs(double[][] trainingData, int[] labels) {
        if (trainingData == null) {
            throw new IllegalArgumentException("trainingData cannot be null");
        }
        if (labels == null) {
            throw new IllegalArgumentException("labels cannot be null");
        }
        if (trainingData.length != labels.length) {
            throw new IllegalArgumentException(
                    String.format("trainingData length (%d) must match labels length (%d)",
                            trainingData.length, labels.length)
            );
        }
        if (trainingData.length == 0) {
            throw new IllegalArgumentException("trainingData cannot be empty");
        }
        
        // Validate that all training samples have the same dimensionality
        int expectedDimension = trainingData[0].length;
        for (int i = 1; i < trainingData.length; i++) {
            if (trainingData[i] == null) {
                throw new IllegalArgumentException("trainingData[" + i + "] cannot be null");
            }
            if (trainingData[i].length != expectedDimension) {
                throw new IllegalArgumentException(
                        String.format("All training samples must have same dimension. Expected %d, got %d at index %d",
                                expectedDimension, trainingData[i].length, i)
                );
            }
        }
    }
    
    /**
     * Validates inputs for the predict method.
     */
    private void validatePredictInputs(int k, double[] testPoint) {
        if (trainingData == null || trainingData.length == 0) {
            throw new IllegalStateException("Classifier must be trained before prediction. Call fit() first.");
        }
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive, got: " + k);
        }
        if (k > trainingData.length) {
            throw new IllegalArgumentException(
                    String.format("k (%d) cannot exceed number of training samples (%d)", k, trainingData.length)
            );
        }
        if (testPoint == null) {
            throw new IllegalArgumentException("testPoint cannot be null");
        }
        if (testPoint.length != trainingData[0].getVector().length) {
            throw new IllegalArgumentException(
                    String.format("testPoint dimension (%d) must match training data dimension (%d)",
                            testPoint.length, trainingData[0].getVector().length)
            );
        }
    }
    
    /**
     * Represents a neighbor with its distance to the test point.
     */
    public static class Neighbor implements Comparable<Neighbor> {
        private final DataPoint trainPoint;
        @Getter
        private final double distance;
        
        public Neighbor(DataPoint trainPoint, double distance) {
            this.trainPoint = trainPoint;
            this.distance = distance;
        }
        
        @Override
        public int compareTo(Neighbor other) {
            return Double.compare(this.distance, other.distance);
        }
        
        public int getLabel() {
            return trainPoint.getLabel();
        }
        
        @Override
        public String toString() {
            return String.format("Neighbor{label=%d, distance=%.4f}", trainPoint.getLabel(), distance);
        }
    }
}