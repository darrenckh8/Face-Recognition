/*
  Face recognition door-lock kiosk enclosure for:
  Elecrow DSI07419T 7 inch 800x480 DSI touch display with bracket.

  Display dimensions are from Elecrow's product page and dimension image:
  https://www.elecrow.com/7-inch-800-480-dsi-display-touch-screen-with-bracket-compatible-with-raspberry-pi.html

  Units: mm

  Suggested use:
  - Render part="assembly" to inspect the complete kiosk.
  - Render part="front_panel" for the printable front/display mount.
  - Render part="rear_shell" for the printable rear enclosure.
  - Render part="display_drill_template" to check the display hole pattern.

  Note:
  Camera mount dimensions default to a Raspberry Pi Camera style module.
  Adjust camera_mount_pitch_x/y if your face-recognition camera differs.
*/

$fn = 64;
eps = 0.02;

part = "assembly"; // "assembly", "front_panel", "rear_shell", "display_drill_template", "display_reference"
part_id = 0;       // CLI override: 1 front, 2 rear, 3 drill template, 4 display ref, 5 assembly
show_reference_parts = true;

// ---------------------------------------------------------------------------
// Elecrow 7 inch DSI display mechanical data
// ---------------------------------------------------------------------------

display_board_w = 164.90;
display_board_h = 102.00;
display_board_t = 12.25;
display_active_w = 153.84;
display_active_h = 85.63;
display_rotation = 90; // Portrait mount. Supported values: 0, 90, 180, 270.
display_inst_board_w = (display_rotation == 90 || display_rotation == 270) ? display_board_h : display_board_w;
display_inst_board_h = (display_rotation == 90 || display_rotation == 270) ? display_board_w : display_board_h;
display_inst_active_w = (display_rotation == 90 || display_rotation == 270) ? display_active_h : display_active_w;
display_inst_active_h = (display_rotation == 90 || display_rotation == 270) ? display_active_w : display_active_h;

// Outer board mount holes: dimension image shows 154.89 x 91.92 mm pitch,
// approximately 5 mm from each board edge. Package screws are M2.5.
display_outer_hole_pitch_x = 154.89;
display_outer_hole_pitch_y = 91.92;
display_outer_hole_inset_x = (display_board_w - display_outer_hole_pitch_x) / 2;
display_outer_hole_inset_y = (display_board_h - display_outer_hole_pitch_y) / 2;

// Inner Raspberry Pi mount pattern shown on the display back.
// The pitch is 58 x 49 mm; x/y offsets below are read from the Elecrow
// dimension image, measured from the display PCB's top-left corner.
display_pi_hole_pitch_x = 58.00;
display_pi_hole_pitch_y = 49.00;
display_pi_hole_right_from_edge = 46.54;
display_pi_hole_top_from_edge = 35.19;
display_pi_hole_x_right = display_board_w - display_pi_hole_right_from_edge;
display_pi_hole_x_left = display_pi_hole_x_right - display_pi_hole_pitch_x;
display_pi_hole_y_top = display_pi_hole_top_from_edge;
display_pi_hole_y_bottom = display_pi_hole_y_top + display_pi_hole_pitch_y;

m2_5_clearance_d = 2.80;
m2_5_pilot_d = 2.20;
m3_clearance_d = 3.30;
m4_clearance_d = 4.50;

// Use clearance holes if you will bolt into inserts/nuts; use m2_5_pilot_d
// if you want plastic bosses for direct self-tapping screws.
display_mount_hole_d = m2_5_clearance_d;
display_boss_outer_d = 8.50;
display_boss_h = 7.00;
display_boss_height_raise = 10.00;
display_boss_lower_cut = 10.00;
display_support_w = 5.50;
display_support_contact_w = 5.50;
display_support_boss_overlap = 1.20;

// ---------------------------------------------------------------------------
// Kiosk body
// ---------------------------------------------------------------------------

body_w = display_inst_board_w + 16;
body_h = display_inst_board_h + 56;
body_d = 45;
corner_r = 8;
wall_t = 3;
front_t = 4;
rear_wall_t = 3;
// Keep the LCD hole pattern and support plane fixed while trimming the lower side.
display_mount_surface_z = front_t + display_boss_h + display_boss_height_raise;
display_standoff_z = front_t + display_boss_lower_cut;
display_standoff_h = display_mount_surface_z - display_standoff_z;

display_center_x = 0;
display_center_y = -7;
display_module_clearance = 1.00;
display_window_clearance = 0.60;
display_window_w = display_inst_board_w + display_module_clearance;
display_window_h = display_inst_board_h + display_module_clearance;
display_visible_w = display_inst_active_w + display_window_clearance;
display_visible_h = display_inst_active_h + display_window_clearance;
display_window_r = 1.25;

// Face recognition camera area.
camera_center_x = 0;
camera_center_y = body_h / 2 - 20;
camera_lens_d = 12.5;
camera_lens_bezel_d = 18;
camera_board_w = 25;
camera_board_h = 24;
camera_mount_pitch_x = 21.0;
camera_mount_pitch_y = 12.5;
camera_mount_hole_d = 2.20;
camera_boss_outer_d = 5.80;
camera_boss_h = 5.00;
camera_mount_bosses_enabled = false;

// Aux front features.
speaker_hole_d = 2.0;
speaker_rows = 3;
speaker_cols = 6;
speaker_pitch = 5;
front_speaker_enabled = false;

// Case screws and wall mounting.
case_screw_inset = 13;
case_screw_boss_d = 9;
case_receiver_h = 10;
case_receiver_hole_d = 2.70; // M3 self-tap pilot; enlarge for heat-set inserts.

// Integrated angled rear housing. The rear shell itself has a 45 degree
// mounting face so it can sit directly against the door frame.
door_frame_angle = 45;
door_frame_side = 1; // -1 left jamb, 1 right jamb.
door_frame_mount_keyhole_spacing = 132;
door_frame_mount_keyhole_head_d = 9.0;
door_frame_mount_keyhole_slot_w = 4.8;
door_frame_mount_keyhole_slot_h = 20;
door_frame_mount_keyhole_depth = 18;

// ---------------------------------------------------------------------------
// Coordinate helpers
// ---------------------------------------------------------------------------

function board_rot_x(x, y) =
    display_rotation == 90 ? -y :
    display_rotation == 180 ? -x :
    display_rotation == 270 ? y :
    x;

function board_rot_y(x, y) =
    display_rotation == 90 ? x :
    display_rotation == 180 ? -y :
    display_rotation == 270 ? -x :
    y;

function board_point(x_from_left, y_from_top) =
    let(
        x = x_from_left - display_board_w / 2,
        y = display_board_h / 2 - y_from_top
    )
    [
        display_center_x + board_rot_x(x, y),
        display_center_y + board_rot_y(x, y)
    ];

display_outer_holes = [
    board_point(display_outer_hole_inset_x, display_outer_hole_inset_y),
    board_point(display_board_w - display_outer_hole_inset_x, display_outer_hole_inset_y),
    board_point(display_outer_hole_inset_x, display_board_h - display_outer_hole_inset_y),
    board_point(display_board_w - display_outer_hole_inset_x, display_board_h - display_outer_hole_inset_y)
];

display_pi_holes = [
    board_point(display_pi_hole_x_left, display_pi_hole_y_top),
    board_point(display_pi_hole_x_right, display_pi_hole_y_top),
    board_point(display_pi_hole_x_left, display_pi_hole_y_bottom),
    board_point(display_pi_hole_x_right, display_pi_hole_y_bottom)
];

case_screw_points = [
    [-body_w / 2 + case_screw_inset, -body_h / 2 + case_screw_inset],
    [ body_w / 2 - case_screw_inset, -body_h / 2 + case_screw_inset],
    [-body_w / 2 + case_screw_inset,  body_h / 2 - case_screw_inset],
    [ body_w / 2 - case_screw_inset,  body_h / 2 - case_screw_inset]
];

// ---------------------------------------------------------------------------
// 2D/3D primitives
// ---------------------------------------------------------------------------

module rounded_rect_2d(w, h, r) {
    rr = min(r, min(w, h) / 2 - 0.01);
    offset(r = rr)
        square([w - 2 * rr, h - 2 * rr], center = true);
}

module rounded_prism(w, h, d, r, z = 0) {
    translate([0, 0, z])
        linear_extrude(height = d)
            rounded_rect_2d(w, h, r);
}

module rounded_cut(w, h, d, r, z = -eps) {
    translate([0, 0, z])
        linear_extrude(height = d + 2 * eps)
            rounded_rect_2d(w, h, r);
}

module round_hole_at(p, d, h, z = -eps) {
    translate([p[0], p[1], z])
        cylinder(d = d, h = h + 2 * eps);
}

module boss_at(p, outer_d, hole_d, h, z) {
    difference() {
        translate([p[0], p[1], z])
            cylinder(d = outer_d, h = h);
        round_hole_at(p, hole_d, h + 2 * eps, z - eps);
    }
}

module horizontal_slot_at(p, w, h, r, depth, z = -eps) {
    translate([p[0], p[1], z])
        linear_extrude(height = depth + 2 * eps)
            rounded_rect_2d(w, h, r);
}

module xz_prism(points, h) {
    rotate([90, 0, 0])
        linear_extrude(height = h, center = true)
            polygon(points = points);
}

module display_mount_tab_at(p) {
    xsign = p[0] < display_center_x ? -1 : 1;
    side_edge_x = xsign * body_w / 2;
    boss_side_x = p[0] + xsign * display_boss_outer_d / 2;
    support_z = front_t - eps;
    support_h = display_mount_surface_z - front_t + 2 * eps;
    contact_z = display_standoff_z - eps;
    contact_h = display_standoff_h + 2 * eps;
    lcd_area_edge_x = display_center_x + xsign * display_window_w / 2;
    support_inner_x = boss_side_x - xsign * display_support_boss_overlap;

    // Main rectangular side support stops at the LCD/window boundary.
    translate([(side_edge_x + lcd_area_edge_x) / 2, p[1], support_z + support_h / 2])
        cube([abs(side_edge_x - lcd_area_edge_x) + eps, display_support_w, support_h], center = true);

    // Only this short contact tab enters the LCD area, just enough to touch
    // the standoff from the side at standoff height.
    translate([(lcd_area_edge_x + support_inner_x) / 2, p[1], contact_z + contact_h / 2])
        cube([abs(lcd_area_edge_x - support_inner_x) + eps, display_support_contact_w, contact_h], center = true);
}

// ---------------------------------------------------------------------------
// Integrated 45 degree rear housing
// ---------------------------------------------------------------------------

function angled_back_z(x) =
    body_d
    + (
        door_frame_side < 0
            ? x + body_w / 2
            : body_w / 2 - x
    ) * tan(door_frame_angle);

function angled_inner_back_z(x) =
    angled_back_z(x) - rear_wall_t / cos(door_frame_angle);

function angled_shell_max_z() =
    max(angled_back_z(-body_w / 2), angled_back_z(body_w / 2));

module angled_back_transform(x = 0, y = 0) {
    translate([x, y, angled_back_z(x)])
        rotate([0, door_frame_side * door_frame_angle, 0])
            children();
}

module angled_back_round_cut(x, y, d, depth) {
    angled_back_transform(x, y)
        translate([0, 0, -depth / 2])
            cylinder(d = d, h = depth);
}

module angled_back_slot_cut(x, y, w, h, depth) {
    angled_back_transform(x, y)
        translate([0, 0, -depth / 2])
            linear_extrude(height = depth)
                rounded_rect_2d(w, h, w / 2);
}

module rear_shell_outer_volume() {
    x0 = -body_w / 2 - eps;
    x1 = body_w / 2 + eps;

    intersection() {
        rounded_prism(
            body_w,
            body_h,
            angled_shell_max_z() - front_t,
            corner_r,
            front_t
        );

        xz_prism(
            [
                [x0, front_t],
                [x1, front_t],
                [x1, angled_back_z(x1)],
                [x0, angled_back_z(x0)]
            ],
            body_h + 2
        );
    }
}

module rear_shell_hollow_cut() {
    x0 = -body_w / 2 + wall_t;
    x1 = body_w / 2 - wall_t;

    intersection() {
        rounded_prism(
            body_w - 2 * wall_t,
            body_h - 2 * wall_t,
            angled_shell_max_z() - front_t + 1,
            corner_r - wall_t,
            front_t - eps
        );

        xz_prism(
            [
                [x0, front_t - eps],
                [x1, front_t - eps],
                [x1, angled_inner_back_z(x1)],
                [x0, angled_inner_back_z(x0)]
            ],
            body_h + 2
        );
    }
}

// ---------------------------------------------------------------------------
// Front panel with display, camera, status, speaker, and bosses
// ---------------------------------------------------------------------------

module front_panel_cutouts() {
    translate([display_center_x, display_center_y, 0])
        rounded_cut(display_window_w, display_window_h, front_t, display_window_r);

    round_hole_at([camera_center_x, camera_center_y], camera_lens_d, front_t);

    if (front_speaker_enabled) {
        // Speaker / buzzer perforations.
        for (ix = [0 : speaker_cols - 1])
            for (iy = [0 : speaker_rows - 1])
                round_hole_at([
                    -body_w / 2 + 31 + ix * speaker_pitch,
                    -body_h / 2 + 27 + iy * speaker_pitch
                ], speaker_hole_d, front_t);
    }

}

module front_panel_bosses() {
    // Display mounting bosses, aligned to Elecrow's outer M2.5 hole pattern.
    for (p = display_outer_holes) {
        display_mount_tab_at(p);
        boss_at(p, display_boss_outer_d, display_mount_hole_d, display_standoff_h, display_standoff_z);
    }

    if (camera_mount_bosses_enabled) {
        // Camera module bosses.
        for (x = [-camera_mount_pitch_x / 2, camera_mount_pitch_x / 2])
            for (y = [-camera_mount_pitch_y / 2, camera_mount_pitch_y / 2])
                boss_at(
                    [camera_center_x + x, camera_center_y + y],
                    camera_boss_outer_d,
                    camera_mount_hole_d,
                    camera_boss_h,
                    front_t
                );
    }

    // Receiver bosses for screws inserted from the rear shell.
    for (p = case_screw_points)
        boss_at(p, case_screw_boss_d, case_receiver_hole_d, case_receiver_h, front_t);
}

module front_panel() {
    union() {
        difference() {
            rounded_prism(body_w, body_h, front_t, corner_r);
            front_panel_cutouts();
        }
        front_panel_bosses();
    }
}

// ---------------------------------------------------------------------------
// Rear shell with integrated 45 degree door-frame mounting face
// ---------------------------------------------------------------------------

module rear_shell_cutouts() {
    // Hollow the angled body, leaving side walls and an angled rear wall.
    rear_shell_hollow_cut();

    // Top ventilation slots through the side wall.
    for (i = [-4 : 4])
        translate([i * 12, body_h / 2 + wall_t / 2, front_t + 20])
            cube([6, wall_t * 4, 26], center = true);

    // Door-frame keyholes cut directly through the angled rear face.
    for (y = [-door_frame_mount_keyhole_spacing / 2, door_frame_mount_keyhole_spacing / 2]) {
        angled_back_round_cut(
            0,
            y,
            door_frame_mount_keyhole_head_d,
            door_frame_mount_keyhole_depth
        );
        angled_back_slot_cut(
            0,
            y + door_frame_mount_keyhole_slot_h / 2 - 2,
            door_frame_mount_keyhole_slot_w,
            door_frame_mount_keyhole_slot_h,
            door_frame_mount_keyhole_depth
        );
    }

    // M3 case screw clearance through the wedge shell, aligned with the
    // front-panel receiver bosses.
    for (p = case_screw_points)
        round_hole_at(p, m3_clearance_d, angled_shell_max_z() - front_t + 2, front_t - eps);
}

module rear_shell_bosses() {
    for (p = case_screw_points)
        boss_at(
            p,
            case_screw_boss_d,
            m3_clearance_d,
            angled_shell_max_z() - front_t + 2,
            front_t
        );
}

module rear_shell() {
    intersection() {
        union() {
            difference() {
                rear_shell_outer_volume();
                rear_shell_cutouts();
            }
            rear_shell_bosses();
        }

        // Clip internal screw sleeves to the angled housing envelope.
        rear_shell_outer_volume();
    }
}

// ---------------------------------------------------------------------------
// Reference parts and templates
// ---------------------------------------------------------------------------

module display_reference() {
    display_back_z = display_mount_surface_z;
    display_front_z = display_back_z - display_board_t;
    display_center_z = (display_back_z + display_front_z) / 2;

    // PCB envelope.
    color([0.02, 0.23, 0.28, 0.35])
        translate([
            display_center_x,
            display_center_y,
            display_center_z
        ])
            rotate([0, 0, display_rotation])
                cube([display_board_w, display_board_h, display_board_t], center = true);

    // Active/touch area.
    color([0.05, 0.55, 0.90, 0.55])
        translate([display_center_x, display_center_y, display_front_z - 0.35])
            rotate([0, 0, display_rotation])
                cube([display_active_w, display_active_h, 0.7], center = true);

    // Outer mount holes.
    color([1, 0.80, 0.05, 0.85])
        for (p = display_outer_holes)
            translate([p[0], p[1], display_back_z + 0.2])
                cylinder(d = 5.2, h = 1.0);

    // Raspberry Pi mount pattern on the display back.
    color([1, 0.25, 0.10, 0.85])
        for (p = display_pi_holes)
            translate([p[0], p[1], display_back_z + 1.6])
                cylinder(d = 4.6, h = 1.0);

    // Approximate Raspberry Pi 4/5 board keep-out mounted to display back.
    color([0.05, 0.42, 0.12, 0.28])
        translate([
            (display_pi_holes[0][0] + display_pi_holes[1][0] + display_pi_holes[2][0] + display_pi_holes[3][0]) / 4 + board_rot_x(3.5, 0),
            (display_pi_holes[0][1] + display_pi_holes[1][1] + display_pi_holes[2][1] + display_pi_holes[3][1]) / 4 + board_rot_y(3.5, 0),
            display_back_z + 9
        ])
            rotate([0, 0, display_rotation])
                cube([85, 56, 17], center = true);
}

module camera_reference() {
    color([0.08, 0.42, 0.12, 0.35])
        translate([camera_center_x, camera_center_y, front_t + camera_boss_h + 0.8])
            cube([camera_board_w, camera_board_h, 1.6], center = true);

    color([0.05, 0.05, 0.05, 0.65])
        translate([camera_center_x, camera_center_y, -1])
            cylinder(d = camera_lens_bezel_d, h = front_t + camera_boss_h + 3);
}

module display_drill_template() {
    difference() {
        union() {
            color([0.85, 0.85, 0.85, 1])
                translate([display_center_x, display_center_y, 0])
                    linear_extrude(height = 2)
                        rotate(display_rotation)
                            square([display_board_w, display_board_h], center = true);

            color([0.10, 0.55, 0.90, 1])
                translate([display_center_x, display_center_y, 2])
                    linear_extrude(height = 0.6)
                        rotate(display_rotation)
                            square([display_active_w, display_active_h], center = true);
        }

        for (p = display_outer_holes)
            round_hole_at(p, display_mount_hole_d, 3);

        for (p = display_pi_holes)
            round_hole_at(p, m2_5_clearance_d, 3);
    }
}

module assembly() {
    union() {
        front_panel();
        rear_shell();
    }

    if (show_reference_parts) {
        display_reference();
        camera_reference();
    }
}

// ---------------------------------------------------------------------------
// Part selector
// ---------------------------------------------------------------------------

if (part_id == 5 || (part_id == 0 && part == "assembly")) {
    assembly();
} else if (part_id == 1 || (part_id == 0 && part == "front_panel")) {
    front_panel();
} else if (part_id == 2 || (part_id == 0 && part == "rear_shell")) {
    rear_shell();
} else if (part_id == 3 || (part_id == 0 && part == "display_drill_template")) {
    display_drill_template();
} else if (part_id == 4 || (part_id == 0 && part == "display_reference")) {
    display_reference();
} else {
    assembly();
}
