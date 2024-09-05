mod base;
mod io;
mod player_controller;
mod random_generation;
mod render;
mod uniform_types;
mod utils;
mod vk_controller;

extern crate nalgebra_glm as glm;

use ash::vk::{self};
use base::AppBase;
use utils::{HEIGHT, WIDTH};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard,
};

#[derive(Default)]
struct App<'a> {
    base: Option<AppBase<'a>>,
}

impl ApplicationHandler for App<'static> {
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let base = self.base.as_mut().unwrap();
        if base.focused {
            base.vk_controller.window.request_redraw();
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.base.is_none() {
            let app = AppBase::new(&event_loop, WIDTH, HEIGHT);

            unsafe {
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                app.vk_controller
                    .device
                    .begin_command_buffer(app.vk_controller.rt_command_buffer, &begin_info)
                    .unwrap();
            }

            unsafe {
                app.vk_controller
                    .device
                    .end_command_buffer(app.vk_controller.rt_command_buffer)
                    .unwrap();
            }

            self.base = Some(app);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let base = self.base.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        logical_key: keyboard::Key::Named(keyboard::NamedKey::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: keyboard::PhysicalKey::Code(keyboard::KeyCode::KeyQ),
                        ..
                    },
                ..
            } => {
                println!("position: {}", base.camera.transform.translation)
            }
            WindowEvent::Resized(_) => self.base.as_mut().unwrap().resized = true,
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => self.base.as_mut().unwrap().toggle_capture_mouse(),
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, y),
                ..
            } => base.player_controller.handle_speed_change(y),
            WindowEvent::KeyboardInput { event, .. } => {
                base.player_controller.handle_keyboard_event(event)
            }
            WindowEvent::RedrawRequested => base.main_loop(),
            _ => (),
        };
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.base.as_mut().unwrap().update_look_position(delta)
            }
            _ => (),
        };
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    let mut app = App::default();

    event_loop.run_app(&mut app).unwrap();
}
