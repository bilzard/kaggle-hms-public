class LinearScheduler:
    """
    初期値から目的の状態まで線形に変化させるスケジューラ。
    (0, initial_value) -> (target_step, target_value)
    """

    def __init__(
        self,
        target_step: int,
        initial_value: float,
        target_value: float,
        schedule_start_step: int = 0,
    ):
        assert (
            schedule_start_step <= target_step
        ), "schedule_start_step should be less than or equal to target_step"
        self.current_step = 0
        self.initial_step = schedule_start_step
        self.current_value = initial_value
        self.target_value = target_value
        self.target_step = target_step

        if self.target_step - self.initial_step > 1:
            self.delta = (target_value - initial_value) / (
                target_step - schedule_start_step
            )
        else:
            self.delta = 0

    def step(self):
        if self.initial_step <= self.current_step < self.target_step:
            self.current_value += self.delta
        elif self.current_step == self.target_step:
            self.current_value = self.target_value
        self.current_step += 1

    @property
    def value(self):
        return self.current_value


if __name__ == "__main__":
    scheduler = LinearScheduler(
        schedule_start_step=5, initial_value=1.0, target_value=0.5, target_step=10
    )

    for epoch in range(15):
        print(f"Epoch {epoch}, Value: {scheduler.value:.4f}")
        scheduler.step()
