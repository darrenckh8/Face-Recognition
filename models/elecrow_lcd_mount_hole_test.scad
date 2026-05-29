/*
  Minimal LCD mounting-hole placement test for the Elecrow DSI07419T
  7 inch 800x480 DSI touch display.

  Units: mm
  Orientation: portrait, matching face_recog_door_lock_kiosk.scad

  Elecrow outer hole pitch from dimension image:
  landscape pitch = 154.89 x 91.92 mm
  portrait pitch  = 91.92 x 154.89 mm
*/

$fn = 48;
eps = 0.02;

hole_pitch_x = 91.92;
hole_pitch_y = 154.89;
hole_d = 2.80;       // M2.5 clearance
pad_d = 11.00;
rib_w = 5.00;
template_t = 2.40;

hole_points = [
    [-hole_pitch_x / 2, -hole_pitch_y / 2],
    [ hole_pitch_x / 2, -hole_pitch_y / 2],
    [-hole_pitch_x / 2,  hole_pitch_y / 2],
    [ hole_pitch_x / 2,  hole_pitch_y / 2]
];

module rounded_rect_2d(w, h, r) {
    rr = min(r, min(w, h) / 2 - 0.01);
    offset(r = rr)
        square([w - 2 * rr, h - 2 * rr], center = true);
}

module pad_at(p) {
    translate([p[0], p[1], 0])
        cylinder(d = pad_d, h = template_t);
}

module hole_at(p) {
    translate([p[0], p[1], -eps])
        cylinder(d = hole_d, h = template_t + 2 * eps);
}

module lcd_mount_hole_test() {
    difference() {
        union() {
            // Four pads.
            for (p = hole_points)
                pad_at(p);

            // Thin ribs: enough to hold spacing, minimal plastic.
            linear_extrude(height = template_t)
                rounded_rect_2d(hole_pitch_x + pad_d, rib_w, rib_w / 2);

            linear_extrude(height = template_t)
                rounded_rect_2d(rib_w, hole_pitch_y + pad_d, rib_w / 2);

            for (y = [-hole_pitch_y / 2, hole_pitch_y / 2])
                translate([0, y, 0])
                    linear_extrude(height = template_t)
                        rounded_rect_2d(hole_pitch_x + pad_d, rib_w, rib_w / 2);
        }

        for (p = hole_points)
            hole_at(p);
    }
}

lcd_mount_hole_test();
