"""MCP-сервер с инструментами анализа отчётов и KPI.

Запуск (stdio):  python -m tools.mcp_server
Сервер ничего не считает сам: регистрирует функции из tools.core, их схемы строятся по аннотациям.
Внимание: по stdio идёт протокол, поэтому в этом процессе нельзя печатать в stdout.
"""
from mcp.server.mcpserver import MCPServer

from tools.core import TOOLS

INSTRUCTIONS = (
    "Инструменты аналитика отчётов и KPI: активный отчёт может быть любой таблицей (дата, разрезы, показатели, план) "
    "или стратегической витриной из трёх листов. Все числа в ответах посчитаны кодом; каждый ответ содержит call_id. "
    "Для вопросов «что изменилось и почему» достаточно diagnose_change; для «что если…» — what_if, для прогноза — forecast_metric."
)


def build_server() -> MCPServer:
    server = MCPServer("bank-kpi-analyst", instructions=INSTRUCTIONS)
    for fn in TOOLS.values():
        server.add_tool(fn)
    return server


if __name__ == "__main__":
    build_server().run(transport="stdio")
