
pub trait Handler<T> {
    fn handle(&mut self, data: T);
}