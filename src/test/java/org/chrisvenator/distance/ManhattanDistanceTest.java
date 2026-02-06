package org.chrisvenator.distance;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

class ManhattanDistanceTest {
    
    private ManhattanDistance manhattanDistance;
    
    @BeforeEach
    void setUp() {
        manhattanDistance = new ManhattanDistance();
    }
    
    @Test
    @DisplayName("Should calculate Manhattan distance between two 1D points")
    void testOneDimensionalDistance() {
        double[] a = {1.0};
        double[] b = {4.0};
        // |4-1| = 3
        assertEquals(3.0, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should calculate Manhattan distance between two 2D points")
    void testTwoDimensionalDistance() {
        double[] a = {0.0, 0.0};
        double[] b = {3.0, 4.0};
        // |3-0| + |4-0| = 3 + 4 = 7
        assertEquals(7.0, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should calculate Manhattan distance between two 3D points")
    void testThreeDimensionalDistance() {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {4.0, 6.0, 8.0};
        // |4-1| + |6-2| + |8-3| = 3 + 4 + 5 = 12
        assertEquals(12.0, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should return 0 for identical points")
    void testIdenticalPoints() {
        double[] a = {2.0, 3.0, 4.0};
        double[] b = {2.0, 3.0, 4.0};
        assertEquals(0.0, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should return 0 for empty arrays")
    void testEmptyArrays() {
        double[] a = {};
        double[] b = {};
        assertEquals(0.0, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should handle negative coordinates")
    void testNegativeCoordinates() {
        double[] a = {-1.0, -2.0};
        double[] b = {2.0, 2.0};
        // |2-(-1)| + |2-(-2)| = 3 + 4 = 7
        assertEquals(7.0, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should handle mixed positive and negative coordinates")
    void testMixedCoordinates() {
        double[] a = {-3.0, 4.0};
        double[] b = {3.0, -4.0};
        // |3-(-3)| + |-4-4| = 6 + 8 = 14
        assertEquals(14.0, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should be symmetric (distance from a to b equals b to a)")
    void testSymmetry() {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {4.0, 5.0, 6.0};
        double distanceAB = manhattanDistance.calculateDistance(a, b);
        double distanceBA = manhattanDistance.calculateDistance(b, a);
        assertEquals(distanceAB, distanceBA, 0.0001);
    }
    
    @Test
    @DisplayName("Should handle high dimensional vectors")
    void testHighDimensionalVectors() {
        double[] a = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] b = {2.0, 3.0, 4.0, 5.0, 6.0};
        // Each dimension differs by 1, so 1+1+1+1+1 = 5
        assertEquals(5.0, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should handle decimal values")
    void testDecimalValues() {
        double[] a = {1.5, 2.5};
        double[] b = {4.0, 6.5};
        // |4-1.5| + |6.5-2.5| = 2.5 + 4.0 = 6.5
        assertEquals(6.5, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should handle fractional differences")
    void testFractionalDifferences() {
        double[] a = {0.1, 0.2, 0.3};
        double[] b = {0.4, 0.5, 0.6};
        // |0.4-0.1| + |0.5-0.2| + |0.6-0.3| = 0.3 + 0.3 + 0.3 = 0.9
        assertEquals(0.9, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should throw exception when first array is null")
    void testFirstArrayNull() {
        double[] b = {1.0, 2.0};
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> manhattanDistance.calculateDistance(null, b));
        assertEquals("a must not be null!", exception.getMessage());
    }
    
    @Test
    @DisplayName("Should throw exception when second array is null")
    void testSecondArrayNull() {
        double[] a = {1.0, 2.0};
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> manhattanDistance.calculateDistance(a, null));
        assertEquals("a must not be null!", exception.getMessage());
    }
    
    @Test
    @DisplayName("Should throw exception when both arrays are null")
    void testBothArraysNull() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> manhattanDistance.calculateDistance(null, null));
        assertEquals("a must not be null!", exception.getMessage());
    }
    
    @Test
    @DisplayName("Should throw exception when arrays have different lengths")
    void testDifferentLengths() {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {1.0, 2.0};
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> manhattanDistance.calculateDistance(a, b));
        assertEquals("a and b length are different!", exception.getMessage());
    }
    
    @Test
    @DisplayName("Should throw exception when first array is longer")
    void testFirstArrayLonger() {
        double[] a = {1.0, 2.0, 3.0, 4.0};
        double[] b = {1.0, 2.0};
        assertThrows(IllegalArgumentException.class, () -> manhattanDistance.calculateDistance(a, b));
    }
    
    @Test
    @DisplayName("Should throw exception when second array is longer")
    void testSecondArrayLonger() {
        double[] a = {1.0, 2.0};
        double[] b = {1.0, 2.0, 3.0, 4.0};
        assertThrows(IllegalArgumentException.class, () -> manhattanDistance.calculateDistance(a, b));
    }
    
    @Test
    @DisplayName("Should handle very large values")
    void testLargeValues() {
        double[] a = {1000.0, 2000.0};
        double[] b = {1100.0, 2200.0};
        // |1100-1000| + |2200-2000| = 100 + 200 = 300
        assertEquals(300.0, manhattanDistance.calculateDistance(a, b), 0.01);
    }
    
    @Test
    @DisplayName("Should handle very small values")
    void testSmallValues() {
        double[] a = {0.001, 0.002};
        double[] b = {0.002, 0.003};
        // |0.002-0.001| + |0.003-0.002| = 0.001 + 0.001 = 0.002
        assertEquals(0.002, manhattanDistance.calculateDistance(a, b), 0.000001);
    }
    
    @Test
    @DisplayName("Should handle zero vectors")
    void testZeroVectors() {
        double[] a = {0.0, 0.0, 0.0};
        double[] b = {0.0, 0.0, 0.0};
        assertEquals(0.0, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should calculate correct distance for unit vectors")
    void testUnitVectors() {
        double[] a = {1.0, 0.0};
        double[] b = {0.0, 1.0};
        // |0-1| + |1-0| = 1 + 1 = 2
        assertEquals(2.0, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should handle all negative values")
    void testAllNegativeValues() {
        double[] a = {-5.0, -3.0};
        double[] b = {-2.0, -1.0};
        // |-2-(-5)| + |-1-(-3)| = 3 + 2 = 5
        assertEquals(5.0, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should verify Manhattan distance is always >= Euclidean distance")
    void testManhattanVsEuclidean() {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {4.0, 6.0, 8.0};
        
        double manhattanDist = manhattanDistance.calculateDistance(a, b);
        // Euclidean would be sqrt(9+16+25) = sqrt(50) ≈ 7.07
        double euclideanDist = Math.sqrt(50);
        
        assertTrue(manhattanDist >= euclideanDist);
    }
    
    @Test
    @DisplayName("Should handle grid-like coordinates")
    void testGridCoordinates() {
        double[] a = {0.0, 0.0};
        double[] b = {5.0, 5.0};
        // Taxi cab distance: |5-0| + |5-0| = 10
        assertEquals(10.0, manhattanDistance.calculateDistance(a, b), 0.0001);
    }
}