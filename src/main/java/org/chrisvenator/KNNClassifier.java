package org.chrisvenator;

import org.chrisvenator.distance.DistanceMetric;

public class KNNClassifier {
    private DataPoint[] trainingData;
    
    public KNNClassifier() {}
    
    /**
     * Train the classifier with data
     *
     * @param trainingData 2D array where each row is a sample, each column a feature
     * @param labels       Array of integer labels for each training sample
     */
    public void fit(double[][] trainingData, int[] labels) {
        if (trainingData == null || labels == null) throw new IllegalArgumentException("trainingData or labels cannot be null!");
        if (trainingData.length != labels.length) throw new IllegalArgumentException("trainingData's and labels' length don't match!");
        
        this.trainingData = new DataPoint[trainingData.length];
        for (int i = 0; i < trainingData.length; i++) {
            this.trainingData[i] = new DataPoint(trainingData[i], labels[i]);
        }
    }
    
    public int predict(int k, double[] testPoint, DistanceMetric distanceMetric) {
        return -1;
    }
}


























