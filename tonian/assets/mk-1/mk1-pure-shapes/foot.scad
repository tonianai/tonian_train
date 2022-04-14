% scale(1000) import("foot.stl");

// Sketch PureShapes 40
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -20.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 40.000000;
translate([0, 0, -thickness]) {
  translate([65.000000, 150.000000, 0]) {
    rotate([0, 0, 180.0]) {
      cube([130.000000, 300.000000, thickness]);
    }
  }
}
}
