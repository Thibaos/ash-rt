use winit::{
    event::KeyEvent,
    keyboard::{Key, NamedKey, SmolStr},
};

pub struct PlayerController {
    pub speed: f32,
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
}

impl Default for PlayerController {
    fn default() -> Self {
        Self {
            speed: 1.0,
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
        }
    }
}

impl PlayerController {
    pub fn handle_speed_change(&mut self, y_delta: f32) {
        if y_delta.is_sign_positive() {
            self.speed *= 1.5;
        } else {
            self.speed /= 1.5;
        }
    }

    pub fn handle_keyboard_event(&mut self, key_event: KeyEvent) {
        let is_pressed = key_event.state.is_pressed();
        if is_pressed {
            if key_event.logical_key == Key::Character(SmolStr::new("z")) {
                self.forward = true;
            };
            if key_event.logical_key == Key::Character(SmolStr::new("s")) {
                self.backward = true;
            }
            if key_event.logical_key == Key::Character(SmolStr::new("q")) {
                self.left = true;
            }
            if key_event.logical_key == Key::Character(SmolStr::new("d")) {
                self.right = true;
            }
            if key_event.logical_key == Key::Named(NamedKey::Space) {
                self.up = true;
            }
            if key_event.logical_key == Key::Named(NamedKey::Control) {
                self.down = true;
            }
        } else {
            if key_event.logical_key == Key::Character(SmolStr::new("z")) {
                self.forward = false;
            };
            if key_event.logical_key == Key::Character(SmolStr::new("s")) {
                self.backward = false;
            }
            if key_event.logical_key == Key::Character(SmolStr::new("q")) {
                self.left = false;
            }
            if key_event.logical_key == Key::Character(SmolStr::new("d")) {
                self.right = false;
            }
            if key_event.logical_key == Key::Named(NamedKey::Space) {
                self.up = false;
            }
            if key_event.logical_key == Key::Named(NamedKey::Control) {
                self.down = false;
            }
        }
    }
}
