package org.chrisvenator;

import lombok.Getter;
import org.chrisvenator.Voting.MajorityVoting;
import org.chrisvenator.Voting.VotingMetric;
import org.chrisvenator.distance.DistanceMetric;
import org.chrisvenator.distance.EuclideanDistance;

import java.util.*;

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
    
    private DataPoint[] shuffleTrainingData() {
        List<DataPoint> list = new ArrayList<>(Arrays.asList(this.trainingData));
        Collections.shuffle(list);
        return list.toArray(new DataPoint[0]);
    }
    
    public Map<Integer, CrossValidationResult> KFoldCrossValidation(int folds, int[] kNeighborsToTest, DistanceMetric distanceMetric, VotingMetric votingMetric) {
        
        if (trainingData == null || trainingData.length == 0) {
            throw new IllegalStateException("Classifier must be trained before cross-validation");
        }
        if (folds <= 1) {
            throw new IllegalArgumentException("folds must be at least 2 for cross-validation");
        }
        if (folds > trainingData.length) {
            throw new IllegalArgumentException("folds cannot exceed number of training samples");
        }
        
        DataPoint[] shuffledData = shuffleTrainingData();
        int foldSize = shuffledData.length / folds;
        
        // Calculate maximum valid k based on smallest training set size
        int maxTrainSize = shuffledData.length - (shuffledData.length - (folds - 1) * foldSize);
        int[] validKValues = Arrays.stream(kNeighborsToTest).filter(k -> k <= maxTrainSize).toArray();
        
        if (validKValues.length == 0) {
            throw new IllegalArgumentException("No valid k values - all exceed training set size");
        }
        
        // Store accuracies for each k value
        Map<Integer, double[]> allAccuracies = new HashMap<>();
        for (int k : validKValues) {
            allAccuracies.put(k, new double[folds]);
        }
        
        for (int i = 0; i < folds; i++) {
            // Calculate validation fold indices
            int validationStart = i * foldSize;
            int validationEnd = (i == folds - 1) ? shuffledData.length : (i + 1) * foldSize;
            
            // Split into training and validation sets
            List<DataPoint> trainList = new ArrayList<>();
            List<DataPoint> validationList = new ArrayList<>();
            
            for (int j = 0; j < shuffledData.length; j++) {
                if (j >= validationStart && j < validationEnd) {
                    validationList.add(shuffledData[j]);
                } else {
                    trainList.add(shuffledData[j]);
                }
            }
            
            // Convert to arrays for training
            double[][] trainData = trainList.stream().map(DataPoint::getVector).toArray(double[][]::new);
            int[] trainLabels = trainList.stream().mapToInt(DataPoint::getLabel).toArray();
            
            // Convert validation data
            double[][] validationData = validationList.stream().map(DataPoint::getVector).toArray(double[][]::new);
            int[] validationLabels = validationList.stream().mapToInt(DataPoint::getLabel).toArray();
            
            // Train once per fold
            KNNClassifier knn = new KNNClassifier();
            knn.fit(trainData, trainLabels);
            
            // Test each k value on this fold
            for (int kNeighbors : validKValues) {
                int correct = 0;
                for (int j = 0; j < validationData.length; j++) {
                    int prediction = knn.predict(kNeighbors, validationData[j], distanceMetric, votingMetric);
                    if (prediction == validationLabels[j]) {
                        correct++;
                    }
                }
                allAccuracies.get(kNeighbors)[i] = (double) correct / validationData.length;
            }
        }
        
        // Convert to CrossValidationResult for each k
        Map<Integer, CrossValidationResult> results = new HashMap<>();
        for (int k : validKValues) {
            results.put(k, new CrossValidationResult(allAccuracies.get(k)));
        }
        
        return results;
    }
    
    public HyperparameterSearchResult findBestHyperparameters(int folds, int[] kValuesToTest, DistanceMetric[] distanceMetrics, VotingMetric[] votingMetrics) {
        double bestAccuracy = -1;
        int bestK = -1;
        DistanceMetric bestDistanceMetric = null;
        VotingMetric bestVotingMetric = null;
        
        Map<String, Map<Integer, CrossValidationResult>> allResults = new HashMap<>();
        for (DistanceMetric distanceMetric : distanceMetrics) {
            for (VotingMetric votingMetric : votingMetrics) {
                String configKey = distanceMetric.toString() + "_" + votingMetric.getClass().getSimpleName();
                Map<Integer, CrossValidationResult> cvResults = KFoldCrossValidation(folds, kValuesToTest, distanceMetric, votingMetric);
                allResults.put(configKey, cvResults);
                
                for (Map.Entry<Integer, CrossValidationResult> entry : cvResults.entrySet()) {
                    double accuracy = entry.getValue().meanAccuracy();
                    if (accuracy > bestAccuracy) {
                        bestAccuracy = accuracy;
                        bestK = entry.getKey();
                        bestDistanceMetric = distanceMetric;
                        bestVotingMetric = votingMetric;
                    }
                }
            }
        }
        return new HyperparameterSearchResult(bestK, bestDistanceMetric, bestVotingMetric, bestAccuracy, allResults);
    }
    
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
            throw new IllegalArgumentException(String.format("trainingData length (%d) must match labels length (%d)", trainingData.length, labels.length));
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
                throw new IllegalArgumentException(String.format("All training samples must have same dimension. Expected %d, got %d at index %d", expectedDimension, trainingData[i].length, i));
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
            throw new IllegalArgumentException(String.format("k (%d) cannot exceed number of training samples (%d)", k, trainingData.length));
        }
        if (testPoint == null) {
            throw new IllegalArgumentException("testPoint cannot be null");
        }
        if (testPoint.length != trainingData[0].getVector().length) {
            throw new IllegalArgumentException(String.format("testPoint dimension (%d) must match training data dimension (%d)", testPoint.length, trainingData[0].getVector().length));
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