import logging
import structlog
from io import StringIO
from arxerr.context import ErrorContext
from arxerr.adapters import log_to_std 

def test_standard_logging_integration():
    # Создаем контекст с логами
    ctx = ErrorContext("test")
    ctx.logs = [
        "[INFO] Starting process",
        "[WARNING] Low resources",
        "[ERROR] Critical failure"
    ]
    
    # Настраиваем стандартное логирование
    stream = StringIO()
    logger = logging.getLogger('arxerr_external_test')
    logger.setLevel(logging.DEBUG)
    
    # Удаляем все существующие обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Передаем логи из контекста
    log_to_std(ctx, logger=logger)
    
    # Проверяем вывод
    output = stream.getvalue()
    
    assert "INFO - Starting process" in output
    assert "WARNING - Low resources" in output
    assert "ERROR - Critical failure" in output

def test_structlog_integration():
    # Настраиваем structlog
    stream = StringIO()
    
    # Конфигурация structlog с добавлением уровня логирования
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        logger_factory=structlog.PrintLoggerFactory(stream)
    )
    logger = structlog.get_logger()
    
    # Создаем контекст с логами
    ctx = ErrorContext("test")
    ctx.logs = [
        "[INFO] System started",
        "[ERROR] Connection timeout"
    ]
    
    # Передаем логи в structlog
    for entry in ctx.logs:
        if entry.startswith("[ERROR]"):
            logger.error(entry[8:])
        else:
            logger.info(entry[7:])
    
    # Проверяем вывод
    output = stream.getvalue()
    
    assert '"event": "System started"' in output
    assert '"event": "Connection timeout"' in output
    assert '"level": "info"' in output
    assert '"level": "error"' in output