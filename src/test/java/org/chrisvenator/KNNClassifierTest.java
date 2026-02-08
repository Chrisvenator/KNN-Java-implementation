package org.chrisvenator;

import org.chrisvenator.Voting.MajorityVoting;
import org.chrisvenator.Voting.WeightedMajorityVoting;
import org.chrisvenator.distance.EuclideanDistance;
import org.chrisvenator.distance.ManhattanDistance;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

class KNNClassifierTest {
    
    private KNNClassifier classifier;
    private double[][] simpleTrainingData;
    private int[] simpleLabels;
    
    @BeforeEach
    void setUp() {
        classifier = new KNNClassifier();
        
        // Simple 2D dataset with 2 classes
        simpleTrainingData = new double[][]{
                {1, 2}, {2, 3}, {3, 1},  // Class 0
                {6, 5}, {7, 7}, {8, 6}   // Class 1
        };
        simpleLabels = new int[]{0, 0, 0, 1, 1, 1};
    }
    
    // ==================== FIT METHOD TESTS ====================
    
    @Test
    @DisplayName("Should throw exception when training data is null")
    void testFitWithNullTrainingData() {
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> classifier.fit(null, simpleLabels)
        );
        assertEquals("trainingData cannot be null", exception.getMessage());
    }
    
    @Test
    @DisplayName("Should throw exception when labels are null")
    void testFitWithNullLabels() {
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> classifier.fit(simpleTrainingData, null)
        );
        assertEquals("labels cannot be null", exception.getMessage());
    }
    
    @Test
    @DisplayName("Should throw exception when lengths don't match")
    void testFitWithMismatchedLengths() {
        int[] wrongLabels = {0, 0, 0};
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> classifier.fit(simpleTrainingData, wrongLabels)
        );
        assertTrue(exception.getMessage().contains("must match"));
    }
    
    @Test
    @DisplayName("Should throw exception when training data is empty")
    void testFitWithEmptyData() {
        double[][] emptyData = new double[0][];
        int[] emptyLabels = new int[0];
        
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> classifier.fit(emptyData, emptyLabels)
        );
        assertEquals("trainingData cannot be empty", exception.getMessage());
    }
    
    @Test
    @DisplayName("Should throw exception when training samples have inconsistent dimensions")
    void testFitWithInconsistentDimensions() {
        double[][] inconsistentData = {
                {1, 2},
                {3, 4},
                {5, 6, 7}  // Different dimension
        };
        int[] labels = {0, 0, 1};
        
        assertThrows(IllegalArgumentException.class, () -> classifier.fit(inconsistentData, labels));
    }
    
    // ==================== PREDICT METHOD TESTS ====================
    
    @Test
    @DisplayName("Should correctly predict class 0 for point near class 0 cluster")
    void testPredictClass0() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {2, 2};
        
        int prediction = classifier.predict(3, testPoint, new EuclideanDistance(), new MajorityVoting());
        assertEquals(0, prediction);
    }
    
    @Test
    @DisplayName("Should correctly predict class 1 for point near class 1 cluster")
    void testPredictClass1() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {7, 6};
        
        int prediction = classifier.predict(3, testPoint, new EuclideanDistance(), new MajorityVoting());
        assertEquals(1, prediction);
    }
    
    @Test
    @DisplayName("Should use default Euclidean distance when distanceMetric is null")
    void testPredictWithNullDistanceMetric() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {2, 2};
        
        assertDoesNotThrow(() -> classifier.predict(3, testPoint, null, new MajorityVoting()));
    }
    
    @Test
    @DisplayName("Should use default MajorityVoting when votingMetric is null")
    void testPredictWithNullVotingMetric() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {2, 2};
        
        assertDoesNotThrow(() -> classifier.predict(3, testPoint, new EuclideanDistance(), null));
    }
    
    @Test
    @DisplayName("Should work with both defaults null")
    void testPredictWithBothDefaultsNull() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {2, 2};
        
        int prediction = classifier.predict(3, testPoint, null, null);
        assertEquals(0, prediction);
    }
    
    @Test
    @DisplayName("Should throw exception when predicting before training")
    void testPredictWithoutTraining() {
        double[] testPoint = {2, 2};
        
        IllegalStateException exception = assertThrows(
                IllegalStateException.class,
                () -> classifier.predict(3, testPoint, new EuclideanDistance(), new MajorityVoting())
        );
        assertTrue(exception.getMessage().contains("must be trained"));
    }
    
    @Test
    @DisplayName("Should throw exception when k is zero")
    void testPredictWithZeroK() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {2, 2};
        
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> classifier.predict(0, testPoint, new EuclideanDistance(), new MajorityVoting())
        );
        assertTrue(exception.getMessage().contains("positive"));
    }
    
    @Test
    @DisplayName("Should throw exception when k is negative")
    void testPredictWithNegativeK() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {2, 2};
        
        assertThrows(IllegalArgumentException.class,
                () -> classifier.predict(-1, testPoint, new EuclideanDistance(), new MajorityVoting())
        );
    }
    
    @Test
    @DisplayName("Should throw exception when k exceeds training size")
    void testPredictWithKTooLarge() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {2, 2};
        
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> classifier.predict(10, testPoint, new EuclideanDistance(), new MajorityVoting())
        );
        assertTrue(exception.getMessage().contains("cannot exceed"));
    }
    
    @Test
    @DisplayName("Should work when k equals training size")
    void testPredictWithKEqualsTrainingSize() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {2, 2};
        
        assertDoesNotThrow(() -> classifier.predict(6, testPoint, new EuclideanDistance(), new MajorityVoting()));
    }
    
    @Test
    @DisplayName("Should throw exception when test point is null")
    void testPredictWithNullTestPoint() {
        classifier.fit(simpleTrainingData, simpleLabels);
        
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> classifier.predict(3, null, new EuclideanDistance(), new MajorityVoting())
        );
        assertEquals("testPoint cannot be null", exception.getMessage());
    }
    
    @Test
    @DisplayName("Should throw exception when test point dimension doesn't match")
    void testPredictWithWrongDimension() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {2, 2, 2};  // 3D instead of 2D
        
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> classifier.predict(3, testPoint, new EuclideanDistance(), new MajorityVoting())
        );
        assertTrue(exception.getMessage().contains("dimension"));
    }
    
    // ==================== DIFFERENT DISTANCE METRICS ====================
    
    @Test
    @DisplayName("Should work with Manhattan distance")
    void testPredictWithManhattanDistance() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {2, 2};
        
        int prediction = classifier.predict(3, testPoint, new ManhattanDistance(), new MajorityVoting());
        assertEquals(0, prediction);
    }
    
    @Test
    @DisplayName("Should work with weighted majority voting")
    void testPredictWithWeightedVoting() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {2, 2};
        
        int prediction = classifier.predict(3, testPoint, new EuclideanDistance(), new WeightedMajorityVoting());
        assertEquals(0, prediction);
    }
    
    // ==================== EDGE CASES ====================
    
    @Test
    @DisplayName("Should work with k=1 (nearest neighbor)")
    void testPredictWithK1() {
        classifier.fit(simpleTrainingData, simpleLabels);
        double[] testPoint = {1.1, 2.1};  // Very close to first training point
        
        int prediction = classifier.predict(1, testPoint, new EuclideanDistance(), new MajorityVoting());
        assertEquals(0, prediction);
    }
    
    @Test
    @DisplayName("Should work with 1D data")
    void testPredictWith1DData() {
        double[][] data1D = {{1}, {2}, {10}, {11}};
        int[] labels1D = {0, 0, 1, 1};
        
        classifier.fit(data1D, labels1D);
        double[] testPoint = {1.5};
        
        int prediction = classifier.predict(2, testPoint, new EuclideanDistance(), new MajorityVoting());
        assertEquals(0, prediction);
    }
    
    @Test
    @DisplayName("Should work with high-dimensional data")
    void testPredictWithHighDimensionalData() {
        double[][] highDimData = {
                {1, 2, 3, 4, 5},
                {2, 3, 4, 5, 6},
                {10, 11, 12, 13, 14},
                {11, 12, 13, 14, 15}
        };
        int[] labels = {0, 0, 1, 1};
        
        classifier.fit(highDimData, labels);
        double[] testPoint = {1.5, 2.5, 3.5, 4.5, 5.5};
        
        int prediction = classifier.predict(2, testPoint, new EuclideanDistance(), new MajorityVoting());
        assertEquals(0, prediction);
    }
    
    @Test
    @DisplayName("Should work with multiple classes")
    void testPredictWithMultipleClasses() {
        double[][] multiClassData = {
                {1, 1}, {2, 2},      // Class 0
                {5, 5}, {6, 6},      // Class 1
                {10, 10}, {11, 11}   // Class 2
        };
        int[] multiClassLabels = {0, 0, 1, 1, 2, 2};
        
        classifier.fit(multiClassData, multiClassLabels);
        
        assertEquals(0, classifier.predict(2, new double[]{1.5, 1.5}, null, null));
        assertEquals(1, classifier.predict(2, new double[]{5.5, 5.5}, null, null));
        assertEquals(2, classifier.predict(2, new double[]{10.5, 10.5}, null, null));
    }
}