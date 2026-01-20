from collections import defaultdict

class LogParser():
    def __init__(self):
        self.logs = defaultdict()

    def add_log(self, log: str):
        parts = log.strip().split()
        date = parts[0]
        time = parts[1]
        level = parts[2].upper()
        msg = "".join(parts[3:])
    
        valid_levels = {"ERROR", "INFO", "WARNING"}
        if level not in valid_levels:
            return
        
        self.logs[level].append(Log(date, time, level, msg))

    
    def count_by_level(self, level) -> int:
        return len(self.logs[level])
    
    def get_errors_by_date(self, date) -> list[str]:
        error_logs = sorted(self.logs["ERROR"], key=lambda log: log.date)
        out = []
        for log in error_logs:
            if date == log.date:
                out.append(log.message)
        return out
