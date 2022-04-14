% scale(1000) import("forearm.stl");

// Sketch PureShapes 140
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -207.00000000000003], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 140.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=40.000000,h=thickness);
  }
}
}
