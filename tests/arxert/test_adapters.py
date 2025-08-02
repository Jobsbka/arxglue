import logging
from io import StringIO
from arxerr.adapters import setup_std_logging, log_to_std

def test_std_logging_adapter():
    # Создаем контекст с логами
    class Context:
        logs = [
            "[INFO] Information message",
            "[WARNING] Warning message",
            "[ERROR] Critical error",
            "Plain message without level"
        ]
    
    # Настраиваем логирование в StringIO
    stream = StringIO()
    logger = logging.getLogger('arxerr_test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Передаем логи из контекста
    log_to_std(Context(), logger=logger)
    
    # Проверяем вывод
    output = stream.getvalue()
    assert "INFO: Information message" in output
    assert "WARNING: Warning message" in output
    assert "ERROR: Critical error" in output
    assert "INFO: Plain message without level" in output