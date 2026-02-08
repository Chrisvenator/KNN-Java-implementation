package org.chrisvenator.Voting;

import org.chrisvenator.DataPoint;
import org.chrisvenator.KNNClassifier;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class WeightedMajorityVotingTest {
    
    private WeightedMajorityVoting voting;
    
    @BeforeEach
    void setUp() {
        voting = new WeightedMajorityVoting();
    }
    
    @Test
    @DisplayName("Should return label with highest weighted vote")
    void testBasicWeightedVoting() {
        List<KNNClassifier.Neighbor> neighbors = new ArrayList<>();
        // Closer neighbor (distance 1) with label 0
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1, 1}, 0), 1.0));
        // Farther neighbor (distance 5) with label 1
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{5, 5}, 1), 5.0));
        // Farther neighbor (distance 6) with label 1
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{6, 6}, 1), 6.0));
        
        // Weights: label 0 = 1/1 = 1.0
        //          label 1 = 1/5 + 1/6 = 0.2 + 0.167 = 0.367
        // Label 0 should win
        int result = voting.vote(3, neighbors);
        assertEquals(0, result);
    }
    
    @Test
    @DisplayName("Should handle identical distances (tie-breaking)")
    void testIdenticalDistances() {
        List<KNNClassifier.Neighbor> neighbors = new ArrayList<>();
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1, 1}, 0), 2.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2, 2}, 1), 2.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{3, 3}, 1), 2.0));
        
        // All weights are equal (1/2), so label 1 should win (2 votes vs 1)
        int result = voting.vote(3, neighbors);
        assertEquals(1, result);
    }
    
    @Test
    @DisplayName("Should handle zero distance (epsilon prevents division by zero)")
    void testZeroDistance() {
        List<KNNClassifier.Neighbor> neighbors = new ArrayList<>();
        // Perfect match (distance 0)
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{0, 0}, 0), 0.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1, 1}, 1), 1.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2, 2}, 1), 2.0));
        
        // Weight for label 0 is ~infinity due to distance 0 + epsilon
        int result = voting.vote(3, neighbors);
        assertEquals(0, result);
    }
    
    @Test
    @DisplayName("Should work with k=1")
    void testSingleNeighbor() {
        List<KNNClassifier.Neighbor> neighbors = new ArrayList<>();
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1, 1}, 5), 1.0));
        
        int result = voting.vote(1, neighbors);
        assertEquals(5, result);
    }
    
    @Test
    @DisplayName("Should only use first k neighbors")
    void testOnlyUsesKNeighbors() {
        List<KNNClassifier.Neighbor> neighbors = new ArrayList<>();
        // First 2 neighbors vote for label 0
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1, 1}, 0), 1.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2, 2}, 0), 1.5));
        // Remaining neighbors vote for label 1 (should be ignored when k=2)
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{3, 3}, 1), 10.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{4, 4}, 1), 11.0));
        
        int result = voting.vote(2, neighbors);
        assertEquals(0, result);
    }
    
    @Test
    @DisplayName("Should give more weight to closer neighbors")
    void testCloserNeighborsHaveMoreWeight() {
        List<KNNClassifier.Neighbor> neighbors = new ArrayList<>();
        // One very close neighbor with label 0 (distance 0.1)
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{0, 0}, 0), 0.1));
        // Two farther neighbors with label 1 (distance 10 each)
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{5, 5}, 1), 10.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{6, 6}, 1), 10.0));
        
        // Weight for label 0: ~10
        // Weight for label 1: 1/10 + 1/10 = 0.2
        // Label 0 should win despite being outnumbered
        int result = voting.vote(3, neighbors);
        assertEquals(0, result);
    }
    
    @Test
    @DisplayName("Should handle multiple classes")
    void testMultipleClasses() {
        List<KNNClassifier.Neighbor> neighbors = new ArrayList<>();
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1, 1}, 0), 1.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2, 2}, 1), 2.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{3, 3}, 2), 3.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{4, 4}, 3), 4.0));
        
        // Label 0 should win (highest weight: 1/1 = 1.0)
        int result = voting.vote(4, neighbors);
        assertEquals(0, result);
    }
    
    // ==================== ERROR HANDLING ====================
    
    @Test
    @DisplayName("Should throw exception for null neighbors list")
    void testNullNeighborsList() {
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> voting.vote(3, null)
        );
        assertTrue(exception.getMessage().contains("null"));
    }
    
    @Test
    @DisplayName("Should throw exception for empty neighbors list")
    void testEmptyNeighborsList() {
        List<KNNClassifier.Neighbor> emptyList = new ArrayList<>();
        
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> voting.vote(3, emptyList)
        );
        assertTrue(exception.getMessage().contains("empty"));
    }
    
    @Test
    @DisplayName("Should throw exception for k <= 0")
    void testInvalidK() {
        List<KNNClassifier.Neighbor> neighbors = new ArrayList<>();
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1, 1}, 0), 1.0));
        
        assertThrows(IllegalArgumentException.class, () -> voting.vote(0, neighbors));
        assertThrows(IllegalArgumentException.class, () -> voting.vote(-1, neighbors));
    }
    
    @Test
    @DisplayName("Should throw exception when k exceeds neighbors size")
    void testKTooLarge() {
        List<KNNClassifier.Neighbor> neighbors = new ArrayList<>();
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1, 1}, 0), 1.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2, 2}, 1), 2.0));
        
        IllegalArgumentException exception = assertThrows(
                IllegalArgumentException.class,
                () -> voting.vote(5, neighbors)
        );
        assertTrue(exception.getMessage().contains("exceed"));
    }
    
    // ==================== COMPARISON WITH MAJORITY VOTING ====================
    
    @Test
    @DisplayName("Should differ from simple majority when distances vary significantly")
    void testDifferenceFromMajorityVoting() {
        List<KNNClassifier.Neighbor> neighbors = new ArrayList<>();
        // One very close neighbor with label 0
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{0, 0}, 0), 0.1));
        // Two far neighbors with label 1
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{5, 5}, 1), 100.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{6, 6}, 1), 100.0));
        
        // Weighted voting should choose 0 (closest)
        WeightedMajorityVoting weightedVoting = new WeightedMajorityVoting();
        int weightedResult = weightedVoting.vote(3, neighbors);
        assertEquals(0, weightedResult);
        
        // Simple majority would choose 1 (2 votes vs 1)
        MajorityVoting majorityVoting = new MajorityVoting();
        int majorityResult = majorityVoting.vote(3, neighbors);
        assertEquals(1, majorityResult);
    }
    
    @Test
    @DisplayName("Should match simple majority when all distances are equal")
    void testMatchesMajorityWhenDistancesEqual() {
        List<KNNClassifier.Neighbor> neighbors = new ArrayList<>();
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1, 1}, 0), 5.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2, 2}, 1), 5.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{3, 3}, 1), 5.0));
        
        WeightedMajorityVoting weightedVoting = new WeightedMajorityVoting();
        MajorityVoting majorityVoting = new MajorityVoting();
        
        int weightedResult = weightedVoting.vote(3, neighbors);
        int majorityResult = majorityVoting.vote(3, neighbors);
        
        // Both should return label 1
        assertEquals(weightedResult, majorityResult);
        assertEquals(1, weightedResult);
    }
}