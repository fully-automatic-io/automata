use tokio::sync::broadcast;

pub struct EventBus<T: Clone + Send + 'static> {
    tx: broadcast::Sender<T>,
}

impl<T: Clone + Send + 'static> EventBus<T> {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<T> {
        self.tx.subscribe()
    }

    pub fn publish(&self, event: T) {
        let _ = self.tx.send(event);
    }

    pub fn sender(&self) -> broadcast::Sender<T> {
        self.tx.clone()
    }
}

impl<T: Clone + Send + 'static> Default for EventBus<T> {
    fn default() -> Self {
        Self::new(64)
    }
}
