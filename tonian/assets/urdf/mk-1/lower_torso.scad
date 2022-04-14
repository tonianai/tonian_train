% scale(1000) import("lower_torso.stl");

// Sketch PureShapes 200
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -100.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 200.000000;
translate([0, 0, -thickness]) {
  translate([-20.000000, -100.000000, 0]) {
    rotate([0, 0, 0.0]) {
      cube([40.000000, 200.000000, thickness]);
    }
  }
}
}
