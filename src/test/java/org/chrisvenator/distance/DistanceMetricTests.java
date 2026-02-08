package org.chrisvenator.distance;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

class MinkowskiDistanceTest {
    
    @Test
    @DisplayName("Should calculate Minkowski distance with p=1 (Manhattan)")
    void testMinkowskiP1() {
        MinkowskiDistance distance = new MinkowskiDistance(1);
        
        double[] a = {0, 0};
        double[] b = {3, 4};
        // Manhattan: |3| + |4| = 7
        assertEquals(7.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should calculate Minkowski distance with p=2 (Euclidean)")
    void testMinkowskiP2() {
        MinkowskiDistance distance = new MinkowskiDistance(2);
        
        double[] a = {0, 0};
        double[] b = {3, 4};
        // Euclidean: sqrt(9 + 16) = 5
        assertEquals(5.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should calculate Minkowski distance with p=3")
    void testMinkowskiP3() {
        MinkowskiDistance distance = new MinkowskiDistance(3);
        
        double[] a = {0, 0};
        double[] b = {2, 2};
        // (2^3 + 2^3)^(1/3) = (16)^(1/3) = 2.52
        assertEquals(Math.pow(16, 1.0/3.0), distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should calculate Minkowski distance with p=10")
    void testMinkowskiP10() {
        MinkowskiDistance distance = new MinkowskiDistance(10);
        
        double[] a = {0, 0};
        double[] b = {1, 1};
        // (1^10 + 1^10)^(1/10) = 2^(1/10)
        assertEquals(Math.pow(2, 0.1), distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should throw exception when p < 1")
    void testMinkowskiInvalidP() {
        assertThrows(IllegalArgumentException.class, () -> new MinkowskiDistance(0.5));
        assertThrows(IllegalArgumentException.class, () -> new MinkowskiDistance(0));
        assertThrows(IllegalArgumentException.class, () -> new MinkowskiDistance(-1));
    }
    
    @Test
    @DisplayName("Should allow p = 1 exactly")
    void testMinkowskiPEquals1() {
        assertDoesNotThrow(() -> new MinkowskiDistance(1.0));
    }
    
    @Test
    @DisplayName("Should handle empty arrays")
    void testMinkowskiEmptyArrays() {
        MinkowskiDistance distance = new MinkowskiDistance(2);
        double[] a = {};
        double[] b = {};
        assertEquals(0.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should throw on null arrays")
    void testMinkowskiNullArrays() {
        MinkowskiDistance distance = new MinkowskiDistance(2);
        double[] a = {1, 2};
        
        assertThrows(IllegalArgumentException.class, () -> distance.calculateDistance(null, a));
        assertThrows(IllegalArgumentException.class, () -> distance.calculateDistance(a, null));
    }
    
    @Test
    @DisplayName("Should throw on different length arrays")
    void testMinkowskiDifferentLengths() {
        MinkowskiDistance distance = new MinkowskiDistance(2);
        double[] a = {1, 2};
        double[] b = {1, 2, 3};
        
        assertThrows(IllegalArgumentException.class, () -> distance.calculateDistance(a, b));
    }
    
    @Test
    @DisplayName("Should be symmetric")
    void testMinkowskiSymmetry() {
        MinkowskiDistance distance = new MinkowskiDistance(3);
        double[] a = {1, 2, 3};
        double[] b = {4, 5, 6};
        
        double distAB = distance.calculateDistance(a, b);
        double distBA = distance.calculateDistance(b, a);
        
        assertEquals(distAB, distBA, 0.0001);
    }
    
    @Test
    @DisplayName("Should return 0 for identical points")
    void testMinkowskiIdenticalPoints() {
        MinkowskiDistance distance = new MinkowskiDistance(5);
        double[] a = {2.5, 3.5, 4.5};
        double[] b = {2.5, 3.5, 4.5};
        
        assertEquals(0.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should handle negative coordinates")
    void testMinkowskiNegativeCoords() {
        MinkowskiDistance distance = new MinkowskiDistance(2);
        double[] a = {-3, -4};
        double[] b = {0, 0};
        
        assertEquals(5.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Verify relationship: higher p gives smaller distance")
    void testMinkowskiPRelationship() {
        double[] a = {0, 0};
        double[] b = {1, 1};
        
        MinkowskiDistance dist1 = new MinkowskiDistance(1);
        MinkowskiDistance dist2 = new MinkowskiDistance(2);
        MinkowskiDistance dist10 = new MinkowskiDistance(10);
        
        double d1 = dist1.calculateDistance(a, b);   // 2.0
        double d2 = dist2.calculateDistance(a, b);   // 1.414...
        double d10 = dist10.calculateDistance(a, b); // 1.072...
        
        assertTrue(d1 > d2);
        assertTrue(d2 > d10);
    }
    
    @Test
    @DisplayName("Should have correct toString")
    void testMinkowskiToString() {
        MinkowskiDistance distance = new MinkowskiDistance(3.5);
        String str = distance.toString();
        
        assertTrue(str.contains("Minkowski"));
        assertTrue(str.contains("3.5") || str.contains("3.50"));
    }
}

class ChebyshevDistanceTest {
    
    @Test
    @DisplayName("Should calculate maximum coordinate difference")
    void testChebyshevBasic() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        double[] a = {1, 2, 3};
        double[] b = {4, 6, 5};
        // Differences: |4-1|=3, |6-2|=4, |5-3|=2
        // Max = 4
        assertEquals(4.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should calculate Chebyshev distance in 2D")
    void testChebyshev2D() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        double[] a = {0, 0};
        double[] b = {3, 4};
        // Max of (3, 4) = 4
        assertEquals(4.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should work when max is in first dimension")
    void testChebyshevMaxFirst() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        double[] a = {0, 0};
        double[] b = {10, 2};
        // Max of (10, 2) = 10
        assertEquals(10.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should work when max is in last dimension")
    void testChebyshevMaxLast() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        double[] a = {0, 0, 0};
        double[] b = {1, 2, 15};
        // Max of (1, 2, 15) = 15
        assertEquals(15.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should return 0 for identical points")
    void testChebyshevIdentical() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        double[] a = {5, 5, 5};
        double[] b = {5, 5, 5};
        assertEquals(0.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should handle empty arrays")
    void testChebyshevEmpty() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        double[] a = {};
        double[] b = {};
        assertEquals(0.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should be symmetric")
    void testChebyshevSymmetry() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        double[] a = {1, 2, 3};
        double[] b = {4, 5, 6};
        
        assertEquals(
                distance.calculateDistance(a, b),
                distance.calculateDistance(b, a),
                0.0001
        );
    }
    
    @Test
    @DisplayName("Should handle negative coordinates")
    void testChebyshevNegative() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        double[] a = {-5, -3};
        double[] b = {2, 1};
        // Differences: |2-(-5)|=7, |1-(-3)|=4
        // Max = 7
        assertEquals(7.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should handle mixed positive/negative")
    void testChebyshevMixed() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        double[] a = {-3, 4};
        double[] b = {3, -4};
        // Differences: |3-(-3)|=6, |-4-4|=8
        // Max = 8
        assertEquals(8.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should throw on null arrays")
    void testChebyshevNull() {
        ChebyshevDistance distance = new ChebyshevDistance();
        double[] a = {1, 2};
        
        assertThrows(IllegalArgumentException.class, () -> distance.calculateDistance(null, a));
        assertThrows(IllegalArgumentException.class, () -> distance.calculateDistance(a, null));
    }
    
    @Test
    @DisplayName("Should throw on different length arrays")
    void testChebyshevDifferentLengths() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        double[] a = {1, 2};
        double[] b = {1, 2, 3};
        
        assertThrows(IllegalArgumentException.class, () -> distance.calculateDistance(a, b));
    }
    
    @Test
    @DisplayName("Should work with 1D arrays")
    void testChebyshev1D() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        double[] a = {5};
        double[] b = {12};
        assertEquals(7.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should handle all equal differences")
    void testChebyshevAllEqual() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        double[] a = {0, 0, 0};
        double[] b = {5, 5, 5};
        // All differences are 5
        assertEquals(5.0, distance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should be <= Manhattan distance")
    void testChebyshevVsManhattan() {
        ChebyshevDistance chebyshev = new ChebyshevDistance();
        ManhattanDistance manhattan = new ManhattanDistance();
        
        double[] a = {1, 2, 3};
        double[] b = {4, 6, 8};
        
        double chebDist = chebyshev.calculateDistance(a, b);
        double manhDist = manhattan.calculateDistance(a, b);
        
        assertTrue(chebDist <= manhDist);
    }
    
    @Test
    @DisplayName("Should be <= Euclidean distance")
    void testChebyshevVsEuclidean() {
        ChebyshevDistance chebyshev = new ChebyshevDistance();
        EuclideanDistance euclidean = new EuclideanDistance();
        
        double[] a = {1, 2, 3};
        double[] b = {4, 6, 8};
        
        double chebDist = chebyshev.calculateDistance(a, b);
        double euclDist = euclidean.calculateDistance(a, b);
        
        assertTrue(chebDist <= euclDist);
    }
    
    @Test
    @DisplayName("Should model chess king movement")
    void testChebyshevChessKing() {
        ChebyshevDistance distance = new ChebyshevDistance();
        
        // Chess board positions
        double[] king = {0, 0};
        double[] target1 = {1, 1};    // Diagonal - 1 move
        double[] target2 = {2, 0};    // Horizontal - 2 moves
        double[] target3 = {3, 2};    // Mixed - 3 moves
        
        assertEquals(1.0, distance.calculateDistance(king, target1), 0.0001);
        assertEquals(2.0, distance.calculateDistance(king, target2), 0.0001);
        assertEquals(3.0, distance.calculateDistance(king, target3), 0.0001);
    }
    
    @Test
    @DisplayName("Should have correct toString")
    void testChebyshevToString() {
        ChebyshevDistance distance = new ChebyshevDistance();
        String str = distance.toString();
        
        assertTrue(str.contains("Chebyshev"));
    }
}