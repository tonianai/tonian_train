% scale(1000) import("upper_arm.stl");

// Sketch PureShapes 420
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 350.00000000000006], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 420.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=50.000000,h=thickness);
  }
}
}

// Sketch PureShapes 330
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 350.00000000000006], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 330.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=50.000000,h=thickness);
  }
}
}
