import math


class GaussianRampUpScheduler:
    """
    schedule_start_step->target_stepまでGauss関数の裾野にしたがって増加するスケジューラ
    """

    def __init__(
        self,
        target_step: int,
        target_value: float,
        schedule_start_step: int = 0,
        **kwargs,
    ):
        self.current_step = 0
        self.target_value = target_value
        self.target_step = target_step
        self.current_value = self.initial_value = self.target_value * math.exp(-5)
        self.schedule_start_step = schedule_start_step

    def step(self):
        if self.current_step < self.schedule_start_step:
            self.current_value = self.initial_value
        elif self.current_step < self.target_step:
            self.current_value = self.target_value * (
                math.exp(
                    -5
                    * (
                        (self.target_step - self.current_step)
                        / (self.target_step - self.schedule_start_step)
                    )
                    ** 2
                )
            )
        else:
            self.current_value = self.target_value
        self.current_step += 1

    @property
    def value(self):
        return self.current_value


if __name__ == "__main__":
    scheduler = GaussianRampUpScheduler(
        target_value=200, target_step=8, schedule_start_step=4
    )

    for epoch in range(16):
        print(f"Epoch {epoch}, Value: {scheduler.value:.02f}")
        scheduler.step()
