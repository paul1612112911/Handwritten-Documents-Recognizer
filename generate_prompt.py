def extract_code_block(file_path: str, target_statements: list[str]):
    import os
    
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"檔案 {file_path} 不存在。")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # 提取所有全域 'from' 和 'import' 指令
    imports = []
    for line in lines:
        if line.startswith('from') or line.startswith('import'):
            imports.append(line)
    
    # 找到目標敘述句和對應程式區塊的索引
    blocks = {}
    found_statements = set()  # 用來追蹤找到的目標敘述句
    for index, line in enumerate(lines):
        stripped_line = line.strip()
        for statement in target_statements:
            if stripped_line.startswith(statement):
                # 記錄已找到的敘述句
                found_statements.add(statement)
                
                # 找到對應的程式區塊
                indent_level = len(line) - len(line.lstrip())  # 目標敘述句的縮排級別
                code_block = [line]
                for subsequent_line in lines[index + 1:]:
                    current_indent_level = len(subsequent_line) - len(subsequent_line.lstrip())
                    
                    # 保留區塊內的空行，並且確保縮排大於目標敘述句的程式區塊
                    if subsequent_line == '\n' or current_indent_level > indent_level:
                        code_block.append(subsequent_line)
                    elif current_indent_level <= indent_level and subsequent_line.strip():  # 遇到相同或更小縮排，且非空行，停止區塊提取
                        break
                blocks[index] = ''.join(code_block)
    
    # 檢查是否所有的目標敘述句都找到
    missing_statements = set(target_statements) - found_statements
    if missing_statements:
        raise ValueError(f"以下敘述句在 '{file_path}' 檔案中找不到: {', '.join(missing_statements)}")
    
    # 根據檔案順序排列 blocks
    sorted_blocks = [blocks[key] for key in sorted(blocks.keys())]
    
    # 將 'from' 和 'import' 指令加到最前面
    full_code = ''.join(imports) + ''.join(sorted_blocks)
    
    return full_code


import pyperclip as pc
def prompt(file_blocks, request):
    result = ''

    intro = '以下是一系列相關的python程式碼：'
    result += intro + '\n'

    for path, statements in file_blocks:
        result += f"{path}\n```\n" + extract_code_block(path, statements).strip() + "\n```\n"
        result += '\n'

    result += request

    pc.copy(result)

    return result

file_blocks = [
    ['nms.py', ['def line_intersect', 'def is_shape_2by2', 'def box_iou_shaped', 'def box_iou_flatten', 'def nms']],
    ['detect_doc.py', ['DEBUG', 'def num_channels', 'def gray', 'def find_docs']],
    ['detect_table.py',['DEBUG', 'def merge_near_num', 'def hough_lines_detect', 'def adaptive_binarize']],
    ['template.py',['DEBUG', 'class Focus', 'def internal_pad_white', 'class Template']],
    ['detect.py', ['DEBUG', 'DIGITS', 'LOWER_LETTERS', 'def draw_bounding_boxes', 'class DocsDetect']],
    ['cls_app.py', ['DEBUG', 'IMG_EXTS', 'get_stem', 'class Result', 'class YoloModelPath', 'class App']],
    ['gui.py',['class MyInputDialog', 'class WindowApp']]
]


print(prompt(file_blocks,'分析這些程式碼'))
# print(extract_code_block(*file_blocks[2]))
