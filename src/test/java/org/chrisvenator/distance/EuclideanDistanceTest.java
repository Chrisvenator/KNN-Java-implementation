package org.chrisvenator.distance;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

class EuclideanDistanceTest {
    
    private EuclideanDistance euclideanDistance;
    
    @BeforeEach
    void setUp() {
        euclideanDistance = new EuclideanDistance();
    }
    
    @Test
    @DisplayName("Should calculate distance between two 1D points")
    void testOneDimensionalDistance() {
        double[] a = {1.0};
        double[] b = {4.0};
        assertEquals(3.0, euclideanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should calculate distance between two 2D points")
    void testTwoDimensionalDistance() {
        double[] a = {0.0, 0.0};
        double[] b = {3.0, 4.0};
        // sqrt((3-0)^2 + (4-0)^2) = sqrt(9 + 16) = 5
        assertEquals(5.0, euclideanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should calculate distance between two 3D points")
    void testThreeDimensionalDistance() {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {4.0, 6.0, 8.0};
        // sqrt((4-1)^2 + (6-2)^2 + (8-3)^2) = sqrt(9 + 16 + 25) = sqrt(50)
        assertEquals(Math.sqrt(50), euclideanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should return 0 for identical points")
    void testIdenticalPoints() {
        double[] a = {2.0, 3.0, 4.0};
        double[] b = {2.0, 3.0, 4.0};
        assertEquals(0.0, euclideanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should return 0 for empty arrays")
    void testEmptyArrays() {
        double[] a = {};
        double[] b = {};
        assertEquals(0.0, euclideanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should handle negative coordinates")
    void testNegativeCoordinates() {
        double[] a = {-1.0, -2.0};
        double[] b = {2.0, 2.0};
        // sqrt((2-(-1))^2 + (2-(-2))^2) = sqrt(9 + 16) = 5
        assertEquals(5.0, euclideanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should handle mixed positive and negative coordinates")
    void testMixedCoordinates() {
        double[] a = {-3.0, 4.0};
        double[] b = {3.0, -4.0};
        // sqrt((3-(-3))^2 + (-4-4)^2) = sqrt(36 + 64) = 10
        assertEquals(10.0, euclideanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should be symmetric (distance from a to b equals b to a)")
    void testSymmetry() {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {4.0, 5.0, 6.0};
        double distanceAB = euclideanDistance.calculateDistance(a, b);
        double distanceBA = euclideanDistance.calculateDistance(b, a);
        assertEquals(distanceAB, distanceBA, 0.0001);
    }
    
    @Test
    @DisplayName("Should handle high dimensional vectors")
    void testHighDimensionalVectors() {
        double[] a = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] b = {2.0, 3.0, 4.0, 5.0, 6.0};
        // Each dimension differs by 1, so sqrt(1+1+1+1+1) = sqrt(5)
        assertEquals(Math.sqrt(5), euclideanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should handle decimal values")
    void testDecimalValues() {
        double[] a = {1.5, 2.5};
        double[] b = {4.0, 6.5};
        // sqrt((4-1.5)^2 + (6.5-2.5)^2) = sqrt(6.25 + 16) = sqrt(22.25)
        assertEquals(Math.sqrt(22.25), euclideanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should throw exception when first array is null")
    void testFirstArrayNull() {
        double[] b = {1.0, 2.0};
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> euclideanDistance.calculateDistance(null, b));
        assertEquals("a must not be null!", exception.getMessage());
    }
    
    @Test
    @DisplayName("Should throw exception when second array is null")
    void testSecondArrayNull() {
        double[] a = {1.0, 2.0};
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> euclideanDistance.calculateDistance(a, null));
        assertEquals("a must not be null!", exception.getMessage());
    }
    
    @Test
    @DisplayName("Should throw exception when both arrays are null")
    void testBothArraysNull() {
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> euclideanDistance.calculateDistance(null, null));
        assertEquals("a must not be null!", exception.getMessage());
    }
    
    @Test
    @DisplayName("Should throw exception when arrays have different lengths")
    void testDifferentLengths() {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {1.0, 2.0};
        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class, () -> euclideanDistance.calculateDistance(a, b));
        assertEquals("a and b length are different!", exception.getMessage());
    }
    
    @Test
    @DisplayName("Should throw exception when first array is longer")
    void testFirstArrayLonger() {
        double[] a = {1.0, 2.0, 3.0, 4.0};
        double[] b = {1.0, 2.0};
        assertThrows(IllegalArgumentException.class, () -> euclideanDistance.calculateDistance(a, b));
    }
    
    @Test
    @DisplayName("Should throw exception when second array is longer")
    void testSecondArrayLonger() {
        double[] a = {1.0, 2.0};
        double[] b = {1.0, 2.0, 3.0, 4.0};
        assertThrows(IllegalArgumentException.class, () -> euclideanDistance.calculateDistance(a, b));
    }
    
    @Test
    @DisplayName("Should handle very large values")
    void testLargeValues() {
        double[] a = {1000.0, 2000.0};
        double[] b = {1100.0, 2200.0};
        // sqrt(100^2 + 200^2) = sqrt(10000 + 40000) = sqrt(50000)
        assertEquals(Math.sqrt(50000), euclideanDistance.calculateDistance(a, b), 0.01);
    }
    
    @Test
    @DisplayName("Should handle very small values")
    void testSmallValues() {
        double[] a = {0.001, 0.002};
        double[] b = {0.002, 0.003};
        // sqrt(0.001^2 + 0.001^2) = sqrt(0.000002)
        assertEquals(Math.sqrt(0.000002), euclideanDistance.calculateDistance(a, b), 0.000001);
    }
    
    @Test
    @DisplayName("Should handle zero vectors")
    void testZeroVectors() {
        double[] a = {0.0, 0.0, 0.0};
        double[] b = {0.0, 0.0, 0.0};
        assertEquals(0.0, euclideanDistance.calculateDistance(a, b), 0.0001);
    }
    
    @Test
    @DisplayName("Should calculate correct distance for unit vectors")
    void testUnitVectors() {
        double[] a = {1.0, 0.0};
        double[] b = {0.0, 1.0};
        // sqrt((0-1)^2 + (1-0)^2) = sqrt(1 + 1) = sqrt(2)
        assertEquals(Math.sqrt(2), euclideanDistance.calculateDistance(a, b), 0.0001);
    }
}