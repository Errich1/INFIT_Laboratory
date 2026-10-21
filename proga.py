import re
import math
from typing import Any


# =====================================================================
# 1. ЛЕКСЕР (ТОКЕНИЗАТОР)
# =====================================================================
# Задача лексера — разбить входную строку на «токены» (мельчайшие
# значимые единицы): числа, операторы, скобки, имена функций и т.д.
# Каждый токен — это кортеж (тип, значение, позиция_в_строке).
# Позиция нужна для понятных сообщений об ошибках.
# =====================================================================
def tokenize(code: str) -> list[tuple[str, str, int]]:
    # Список правил: (имя_типа_токена, регулярное_выражение).
    # ВАЖЕН ПОРЯДОК: правила проверяются сверху вниз.
    token_specification = [
        ('EQ',      r'=='),                    # оператор сравнения
        ('FUNC',    r'\bceil\b'),              # имя функции ceil (целое слово)
        ('NUMBER',  r'\d+(\.\d*)?'),           # целые и дробные числа (12, 3.14, 5.)
        ('OP',      r'[+\-*/]'),               # арифметические операторы
        ('LPAREN',  r'\('),                    # открывающая скобка
        ('RPAREN',  r'\)'),                    # закрывающая скобка
        ('SKIP',    r'[ \t\n\r]+'),            # пробелы — игнорируем
        ('MISMATCH', r'.'),                    # любой другой символ = ошибка
    ]

    # Склеиваем правила в одну регулярку с именованными группами.
    # Пример результата: '(?P<EQ>==)|(?P<FUNC>\bceil\b)|...'
    tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)

    tokens = []

    # finditer идёт по строке слева направо и на каждом шаге
    # пытается применить правила; срабатывает первое подходящее.
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup   # имя сработавшей группы = тип токена
        value = mo.group(kind) # сам текст токена
        pos = mo.start()       # позиция в исходной строке

        if kind in ('EQ', 'FUNC', 'NUMBER', 'OP', 'LPAREN', 'RPAREN'):
            # Значимые токены — сохраняем
            tokens.append((kind, value, pos))
        elif kind == 'SKIP':
            # Пробелы и переводы строк просто пропускаем
            continue
        elif kind == 'MISMATCH':
            # Встретили символ, которого нет ни в одном правиле
            raise SyntaxError(f"Неизвестный символ '{value}' на позиции {pos}")

    # EOF (End Of File) — искусственный токен-«стоп-сигнал».
    # Он упрощает парсер: там, где ожидается конец выражения,
    # достаточно проверить, что текущий токен имеет тип 'EOF'.
    tokens.append(('EOF', '', len(code)))
    return tokens


# =====================================================================
# 2. ПАРСЕР (СИНТАКСИЧЕСКИЙ АНАЛИЗАТОР)
# =====================================================================
# Парсер строит из плоского списка токенов дерево разбора (AST —
# Abstract Syntax Tree). Используется метод рекурсивного спуска:
# у каждой «ступени» приоритета операций своя функция.
# Каждая функция возвращает пару: (построенный_узел, индекс_следующего_токена).
def parse(tokens: list[tuple[str, str, int]]) -> dict:
    # Начинаем разбор с самого «низкоприоритетного» уровня — равенства.
    ast, index = parse_equality(tokens, 0)

    # После разбора всех токенов должен остаться только EOF,
    # иначе пользователь написал что-то лишнее.
    if tokens[index][0] != 'EOF':
        raise SyntaxError(f"Лишние символы в выражении на позиции {tokens[index][2]}")
    return ast


def parse_equality(tokens: list, index: int) -> tuple[dict, int]:
    # Левое выражение — всё, что может стоять слева от '=='
    node, index = parse_expr(tokens, index)

    # Если дальше идёт '==' — строим бинарный узел.
    # Цикл while позволяет обрабатывать цепочки "a == b == c".
    while tokens[index][0] == 'EQ':
        right_node, next_index = parse_expr(tokens, index + 1)
        node = {'type': 'binop', 'op': '==', 'left': node, 'right': right_node}
        index = next_index
    return node, index


def parse_expr(tokens: list, index: int) -> tuple[dict, int]:
    # Уровень сложения/вычитания.
    # Сначала разбираем более приоритетный «терм» (умножение/деление),
    # а потом, пока видим '+' или '-', строим узлы.
    node, index = parse_term(tokens, index)
    while tokens[index][0] == 'OP' and tokens[index][1] in ('+', '-'):
        op = tokens[index][1]
        right_node, next_index = parse_term(tokens, index + 1)
        node = {'type': 'binop', 'op': op, 'left': node, 'right': right_node}
        index = next_index
    return node, index


def parse_term(tokens: list, index: int) -> tuple[dict, int]:
    # Уровень умножения/деления.
    # Сначала разбираем «фактор», потом «навешиваем» операции '*' и '/'.
    node, index = parse_factor(tokens, index)
    while tokens[index][0] == 'OP' and tokens[index][1] in ('*', '/'):
        op = tokens[index][1]
        right_node, next_index = parse_factor(tokens, index + 1)
        node = {'type': 'binop', 'op': op, 'left': node, 'right': right_node}
        index = next_index
    return node, index


def parse_factor(tokens: list, index: int) -> tuple[dict, int]:
    # Самый высокий уровень приоритета: числа, унарный минус,
    # вызовы функций, выражения в скобках.
    token_type, token_value, pos = tokens[index]

    # 1) Обработка УНАРНОГО МИНУСА.
    # Если встречаем '-', это может быть как бинарное вычитание (a - b),
    # так и унарный минус (-a). На этом уровне мы точно знаем, что
    # если оператор '-' стоит в начале, это унарный минус.
    # Мы рекурсивно вызываем parse_factor, чтобы получить операнд (например, число).
    if token_type == 'OP' and token_value == '-':
        operand_node, next_index = parse_factor(tokens, index + 1)
        # Возвращаем специальный узел 'unary', который интерпретатор
        # обработает, поменяв знак операнда.
        return {'type': 'unary', 'op': '-', 'operand': operand_node}, next_index

    # 2) Число — просто узел-литерал. Приводим к float.
    if token_type == 'NUMBER':
        return {'type': 'number', 'value': float(token_value)}, index + 1

    # 3) Вызов функции: FUNC '(' выражение ')'
    if token_type == 'FUNC':
        func_name = token_value
        index += 1
        if tokens[index][0] != 'LPAREN':
            raise SyntaxError(f"Ожидалась '(' после функции {func_name} на позиции {tokens[index][2]}")

        # Аргумент функции — это полноценное выражение.
        arg_node, next_index = parse_equality(tokens, index + 1)

        if tokens[next_index][0] != 'RPAREN':
            raise SyntaxError(f"Ожидалась закрывающая скобка ')' на позиции {tokens[next_index][2]}")

        return {'type': 'call', 'func': func_name, 'arg': arg_node}, next_index + 1

    # 4) Скобки: '(' выражение ')' — возвращаем внутренний узел.
    if token_type == 'LPAREN':
        node, next_index = parse_equality(tokens, index + 1)
        if tokens[next_index][0] != 'RPAREN':
            raise SyntaxError(f"Ожидалась закрывающая скобка на позиции {tokens[next_index][2]}")
        return node, next_index + 1

    # 5) Всё остальное — синтаксическая ошибка.
    raise SyntaxError(f"Ожидалось число или выражение на позиции {pos}")


# =====================================================================
# 3. ИНТЕРПРЕТАТОР (ВЫЧИСЛИТЕЛЬ AST)
# =====================================================================
# Обходит построенное дерево рекурсивно и возвращает значение.
# =====================================================================
def evaluate(node: dict) -> Any:
    # Литерал — просто возвращаем значение.
    if node['type'] == 'number':
        return node['value']

    # Обработка УНАРНОГО МИНУСА.
    # Вычисляем операнд и возвращаем его с противоположным знаком.
    if node['type'] == 'unary':
        operand = evaluate(node['operand'])
        if node['op'] == '-': return -operand

    # Бинарная операция: сначала вычисляем обе ветви, потом применяем op.
    if node['type'] == 'binop':
        left = evaluate(node['left'])
        right = evaluate(node['right'])
        op = node['op']

        if op == '+': return left + right
        if op == '-': return left - right
        if op == '*': return left * right
        if op == '/':
            if right == 0:
                raise ZeroDivisionError("Ошибка: Деление на ноль!")
            return left / right
        if op == '==': return left == right   # вернёт True/False

    # Вызов функции ceil.
    if node['type'] == 'call':
        arg = evaluate(node['arg'])

        # Если внутри получилось True/False (например, ceil(1==1)),
        # приводим к float, чтобы math.ceil не упал на bool.
        if isinstance(arg, bool):
            arg = float(arg)

        if node['func'] == 'ceil': return float(math.ceil(arg))

    # Страховка от неизвестных узлов.
    raise ValueError(f"Неизвестный тип узла AST: {node['type']}")


# =====================================================================
# 4. ИНТЕРАКТИВНЫЙ ИНТЕРФЕЙС (REPL)
# =====================================================================
# Read-Eval-Print Loop: читаем строку → лексер → парсер → интерпретатор →
# печатаем результат. Исключения перехватываются, чтобы программа
# не падала от опечатки пользователя.
# =====================================================================
if __name__ == '__main__':
    print("=== Функциональный Калькулятор (Лексер + Парсер) ===")
    print("Поддерживает: +, -, *, /, ==, унарный минус, скобки (), функцию ceil()")

    while True:
        try:
            # 1) Читаем ввод и убираем лишние пробелы по краям.
            user_input = input("Ввод > ").strip()

            # 2) Цепочка обработки: строка → токены → AST → значение.
            tokens = tokenize(user_input)
            ast_tree = parse(tokens)
            result = evaluate(ast_tree)

            # 3) Печатаем ответ.
            print(f"Ответ: {result}\n")

        except SyntaxError as e:
            # Ошибки лексера/парсера (неизвестный символ, пропущена скобка…)
            print(f"Ошибка синтаксиса: {e}\n")
        except ZeroDivisionError as e:
            # Деление на ноль — отдельный тип исключения.
            print(f"{e}\n")
        except Exception as e:
            # Страховка от любых непредвиденных ошибок.
            print(f"Непредвиденная ошибка: {e}\n")