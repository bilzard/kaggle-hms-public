class LinearScheduler:
    """
    初期値から目的の状態まで線形に変化させるスケジューラ。
    (0, initial_value) -> (target_step, target_value)
    """

    def __init__(self, initial_value, target_value, target_step):
        self.current_step = 0
        self._current_value = initial_value
        self.target_value = target_value
        self.target_step = target_step

        if self.target_step > 1:
            self.delta = (target_value - initial_value) / target_step
        else:
            self.delta = 0

    def step(self):
        if self.current_step < self.target_step:
            self._current_value += self.delta
        self.current_step += 1

    @property
    def value(self):
        return self._current_value


if __name__ == "__main__":
    scheduler = LinearScheduler(initial_value=1.0, target_value=0.5, target_step=10)

    for epoch in range(15):
        print(f"Epoch {epoch}, Value: {scheduler.value:.4f}")
        scheduler.step()
