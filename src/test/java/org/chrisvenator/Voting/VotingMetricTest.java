package org.chrisvenator.Voting;

import org.chrisvenator.DataPoint;
import org.chrisvenator.KNNClassifier;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class VotingMetricTest {
    
    private List<KNNClassifier.Neighbor> neighbors;
    
    @BeforeEach
    void setUp() {
        // Create a list of neighbors for testing
        neighbors = new ArrayList<>();
        // Closer neighbors first (sorted by distance)
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1, 1}, 0), 1.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2, 2}, 0), 2.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{3, 3}, 1), 3.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{4, 4}, 1), 4.0));
        neighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{5, 5}, 1), 5.0));
    }
    
    // ==================== MAJORITY VOTING TESTS ====================
    
    @Test
    @DisplayName("MajorityVoting: Should return most common label")
    void testMajorityVotingBasic() {
        MajorityVoting voting = new MajorityVoting();
        
        // k=3: labels are [0, 0, 1] -> majority is 0
        int result = voting.vote(3, neighbors);
        assertEquals(0, result);
        
        // k=5: labels are [0, 0, 1, 1, 1] -> majority is 1
        result = voting.vote(5, neighbors);
        assertEquals(1, result);
    }
    
    @Test
    @DisplayName("MajorityVoting: Should handle k=1")
    void testMajorityVotingK1() {
        MajorityVoting voting = new MajorityVoting();
        
        int result = voting.vote(1, neighbors);
        assertEquals(0, result);  // First neighbor has label 0
    }
    
    @Test
    @DisplayName("MajorityVoting: Should handle ties (returns first in max)")
    void testMajorityVotingTie() {
        MajorityVoting voting = new MajorityVoting();
        
        // k=4: labels are [0, 0, 1, 1] -> tie
        int result = voting.vote(4, neighbors);
        assertTrue(result == 0 || result == 1);  // Either is valid for a tie
    }
    
    @Test
    @DisplayName("MajorityVoting: Should handle all same label")
    void testMajorityVotingAllSame() {
        MajorityVoting voting = new MajorityVoting();
        
        List<KNNClassifier.Neighbor> sameLabel = new ArrayList<>();
        sameLabel.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1}, 5), 1.0));
        sameLabel.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2}, 5), 2.0));
        sameLabel.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{3}, 5), 3.0));
        
        int result = voting.vote(3, sameLabel);
        assertEquals(5, result);
    }
    
    // ==================== WEIGHTED MAJORITY VOTING TESTS ====================
    
    @Test
    @DisplayName("WeightedVoting: Should weight closer neighbors more heavily")
    void testWeightedVotingBasic() {
        WeightedMajorityVoting voting = new WeightedMajorityVoting();
        
        // Create neighbors where class 1 is more common but farther
        List<KNNClassifier.Neighbor> testNeighbors = new ArrayList<>();
        testNeighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1}, 0), 1.0));  // weight = 1/1 = 1.0
        testNeighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2}, 1), 5.0));  // weight = 1/5 = 0.2
        testNeighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{3}, 1), 6.0));  // weight = 1/6 ≈ 0.17
        
        // Class 0 has weight 1.0, class 1 has weight 0.2 + 0.17 = 0.37
        // So class 0 should win despite being minority
        int result = voting.vote(3, testNeighbors);
        assertEquals(0, result);
    }
    
    @Test
    @DisplayName("WeightedVoting: Should handle distance = 0 (exact match)")
    void testWeightedVotingExactMatch() {
        WeightedMajorityVoting voting = new WeightedMajorityVoting();
        
        List<KNNClassifier.Neighbor> testNeighbors = new ArrayList<>();
        testNeighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1}, 0), 0.0));  // Exact match!
        testNeighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2}, 1), 1.0));
        testNeighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{3}, 1), 1.0));
        
        // Distance 0 should have very high weight, so class 0 wins
        int result = voting.vote(3, testNeighbors);
        assertEquals(0, result);
    }
    
    @Test
    @DisplayName("WeightedVoting: Should handle k=1")
    void testWeightedVotingK1() {
        WeightedMajorityVoting voting = new WeightedMajorityVoting();
        
        int result = voting.vote(1, neighbors);
        assertEquals(0, result);
    }
    
    @Test
    @DisplayName("WeightedVoting: Should handle all equal distances")
    void testWeightedVotingEqualDistances() {
        WeightedMajorityVoting voting = new WeightedMajorityVoting();
        
        List<KNNClassifier.Neighbor> equalDist = new ArrayList<>();
        equalDist.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1}, 0), 5.0));
        equalDist.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2}, 0), 5.0));
        equalDist.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{3}, 1), 5.0));
        
        // Equal weights, so it becomes like majority voting
        int result = voting.vote(3, equalDist);
        assertEquals(0, result);  // 2 vs 1
    }
    
    @Test
    @DisplayName("WeightedVoting: Should throw on k <= 0")
    void testWeightedVotingInvalidK() {
        WeightedMajorityVoting voting = new WeightedMajorityVoting();
        
        assertThrows(IllegalArgumentException.class, () -> voting.vote(0, neighbors));
        assertThrows(IllegalArgumentException.class, () -> voting.vote(-1, neighbors));
    }
    
    @Test
    @DisplayName("WeightedVoting: Should throw on null neighbors")
    void testWeightedVotingNullNeighbors() {
        WeightedMajorityVoting voting = new WeightedMajorityVoting();
        
        assertThrows(IllegalArgumentException.class, () -> voting.vote(1, null));
    }
    
    @Test
    @DisplayName("WeightedVoting: Should throw on k > neighbors.size()")
    void testWeightedVotingKTooLarge() {
        WeightedMajorityVoting voting = new WeightedMajorityVoting();
        
        assertThrows(IllegalArgumentException.class, () -> voting.vote(10, neighbors));
    }
    
    @Test
    @DisplayName("WeightedVoting: Should handle multiple classes")
    void testWeightedVotingMultipleClasses() {
        WeightedMajorityVoting voting = new WeightedMajorityVoting();
        
        List<KNNClassifier.Neighbor> multiClass = new ArrayList<>();
        multiClass.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1}, 0), 1.0));
        multiClass.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2}, 1), 2.0));
        multiClass.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{3}, 2), 3.0));
        multiClass.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{4}, 3), 4.0));
        
        int result = voting.vote(4, multiClass);
        // Class 0 has highest weight (1.0), so it should win
        assertEquals(0, result);
    }
    
    // ==================== COMPARISON TESTS ====================
    
    @Test
    @DisplayName("Comparison: Weighted should differ from Majority when distances vary")
    void testWeightedVsMajority() {
        MajorityVoting majority = new MajorityVoting();
        WeightedMajorityVoting weighted = new WeightedMajorityVoting();
        
        // Create scenario where majority and weighted voting differ
        List<KNNClassifier.Neighbor> testNeighbors = new ArrayList<>();
        testNeighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1}, 0), 1.0));
        testNeighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2}, 1), 10.0));
        testNeighbors.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{3}, 1), 11.0));
        
        int majorityResult = majority.vote(3, testNeighbors);
        int weightedResult = weighted.vote(3, testNeighbors);
        
        // Majority: 1 wins (2 vs 1)
        assertEquals(1, majorityResult);
        
        // Weighted: 0 should win (weight 1.0 vs 0.1+0.09)
        assertEquals(0, weightedResult);
    }
    
    @Test
    @DisplayName("Comparison: Should agree when distances are equal")
    void testAgreementOnEqualDistances() {
        MajorityVoting majority = new MajorityVoting();
        WeightedMajorityVoting weighted = new WeightedMajorityVoting();
        
        List<KNNClassifier.Neighbor> equalDist = new ArrayList<>();
        equalDist.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{1}, 0), 5.0));
        equalDist.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{2}, 0), 5.0));
        equalDist.add(new KNNClassifier.Neighbor(new DataPoint(new double[]{3}, 1), 5.0));
        
        int majorityResult = majority.vote(3, equalDist);
        int weightedResult = weighted.vote(3, equalDist);
        
        // Should agree when all weights are equal
        assertEquals(majorityResult, weightedResult);
    }
}