package org.chrisvenator;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.chrisvenator.Voting.MajorityVoting;
import org.chrisvenator.Voting.VotingMetric;
import org.chrisvenator.distance.DistanceMetric;
import org.chrisvenator.distance.EuclideanDistance;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

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
    
    public int predict(int k, double[] testPoint, DistanceMetric distanceMetric, VotingMetric votingMetric) {
        if (distanceMetric == null) distanceMetric = new EuclideanDistance();
        if (votingMetric == null) votingMetric = new MajorityVoting();
        
        List<Neighbor> space = new ArrayList<>();
        for (DataPoint dataPoint : trainingData) {
            space.add(new Neighbor(dataPoint, distanceMetric.calculateDistance(dataPoint.getVector(), testPoint)));
        }
        space.sort(Comparator.comparingDouble(Neighbor::getDistance));
        
        return votingMetric.vote(k, space);
    }
    
    @AllArgsConstructor
    public static class Neighbor implements Comparable<Neighbor> {
        private DataPoint trainPoint;
        @Getter
        private double distance;
        
        @Override
        public int compareTo(Neighbor other) {
            return Double.compare(this.distance, other.distance);
        }
        
        public int getLabel() {
            return trainPoint.getLabel();
        }
    }
}


























