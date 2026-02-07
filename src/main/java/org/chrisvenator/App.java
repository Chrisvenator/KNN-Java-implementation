package org.chrisvenator;

import org.chrisvenator.Voting.MajorityVoting;
import org.chrisvenator.distance.EuclideanDistance;

/**
 * Hello world!
 */
public class App {
    public static void main(String[] args) {
        double[][] trainingData = {{1, 2}, {2, 3}, {3, 1}, {6, 5}, {7, 7}, {8, 6}};
        int[] labels = {0, 0, 0, 1, 1, 1};
        
        KNNClassifier knn = new KNNClassifier();
        knn.fit(trainingData, labels);
        
        double[] test = {2, 2};
        System.out.println(knn.predict(6, test, new EuclideanDistance(), new MajorityVoting()));
    }
}
