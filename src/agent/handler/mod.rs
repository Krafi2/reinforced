pub mod replay_buffer;

pub trait Handler<T> {
    fn handle(&mut self, message: T);
}
