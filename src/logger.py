class BaseLogger:
    def __init__(self, log_file_name: str, clear: bool = True):
        self.log_file_name = log_file_name
        if clear:
            self.clear()

    def write_log(self, *args, sep=" ", end="\n"):
        message = sep.join(map(str, args)) + end
        with open(self.log_file_name, "a") as fp:
            fp.write(message)

    def clear(self):
        with open(self.log_file_name, "w"):
            pass
