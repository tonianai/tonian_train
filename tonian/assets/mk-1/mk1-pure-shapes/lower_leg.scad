% scale(1000) import("lower_leg.stl");

// Sketch PureShapes 100
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -100.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 100.000000;
translate([0, 0, -thickness]) {
  translate([-70.000000, -180.000000, 0]) {
    rotate([0, 0, 0.0]) {
      cube([140.000000, 430.000000, thickness]);
    }
  }
}
}
