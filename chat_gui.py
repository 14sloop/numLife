import os
import sys
import random
import argparse
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QTextEdit, QListWidget, QStackedWidget,
                           QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
                           QScrollArea, QFrame, QSizePolicy, QMessageBox)
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QIcon, QTextCursor, QColor, QTextCharFormat, QTextFormat
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import numlifeLM
from model.LMConfig import LMConfig
from model.model_lora import apply_lora, load_lora

# 自定义信号类
class ModelLoaderSignals(QThread):
    model_loaded = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

# 修改 ModelLoader 类
class ModelLoader(QThread):
    # 添加信号定义
    model_loaded = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
    
    def run(self):
        try:
            if self.args.load == 0:
                moe_path = '_moe' if self.args.use_moe else ''
                modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
                ckp = f'./{self.args.out_dir}/{modes[self.args.model_mode]}_{self.args.dim}{moe_path}.pth'

                model = numlifeLM(LMConfig(
                    dim=self.args.dim,
                    n_layers=self.args.n_layers,
                    max_seq_len=self.args.max_seq_len,
                    use_moe=self.args.use_moe
                ))

                state_dict = torch.load(ckp, map_location=self.args.device)
                model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

                if self.args.lora_name != 'None':
                    apply_lora(model)
                    load_lora(model, f'./{self.args.out_dir}/lora/{self.args.lora_name}_{self.args.dim}.pth')
            else:
                project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                hf_model_path = project_root / "numlife2"
                
                model = AutoModelForCausalLM.from_pretrained(str(hf_model_path), trust_remote_code=True)
            
            model.eval().to(self.args.device)
            self.model_loaded.emit(model)
        except Exception as e:
            print(f"加载模型出错: {str(e)}")  # 添加调试信息
            self.error_occurred.emit(str(e))

class ModelSelectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("numlife - 模型选择")
        self.setFixedSize(800, 600)
        self.init_ui()
        
    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 标题
        title = QLabel("numlife 模型配置")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # 配置区域
        config_frame = QFrame()
        config_frame.setFrameShape(QFrame.StyledPanel)
        config_layout = QVBoxLayout()
        config_frame.setLayout(config_layout)
        
        # 模型模式选择
        model_mode_layout = QHBoxLayout()
        model_mode_label = QLabel("模型模式:")
        model_mode_label.setFont(QFont("Arial", 12))
        self.model_mode_combo = QComboBox()
        self.model_mode_combo.addItems(["预训练模型", "SFT-Chat模型", "RLHF-Chat模型", "Reason模型"])
        self.model_mode_combo.setCurrentIndex(2)  # 默认RLHF-Chat
        model_mode_layout.addWidget(model_mode_label)
        model_mode_layout.addWidget(self.model_mode_combo)
        config_layout.addLayout(model_mode_layout)
        
        # 加载方式选择
        load_mode_layout = QHBoxLayout()
        load_mode_label = QLabel("加载方式:")
        load_mode_label.setFont(QFont("Arial", 12))
        self.load_mode_combo = QComboBox()
        self.load_mode_combo.addItems(["原生torch权重", "HuggingFace模型"])
        self.load_mode_combo.setCurrentIndex(1)  # 默认HuggingFace
        load_mode_layout.addWidget(load_mode_label)
        load_mode_layout.addWidget(self.load_mode_combo)
        config_layout.addLayout(load_mode_layout)
        
        # 高级选项按钮
        self.advanced_btn = QPushButton("高级选项 ▼")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.setChecked(False)
        self.advanced_btn.clicked.connect(self.toggle_advanced)
        config_layout.addWidget(self.advanced_btn)
        
        # 高级选项区域 (默认隐藏)
        self.advanced_frame = QFrame()
        self.advanced_frame.setVisible(False)
        advanced_layout = QVBoxLayout()
        self.advanced_frame.setLayout(advanced_layout)
        
        # LoRA选项
        lora_layout = QHBoxLayout()
        lora_label = QLabel("LoRA适配器:")
        lora_label.setFont(QFont("Arial", 10))
        self.lora_combo = QComboBox()
        self.lora_combo.addItems(["None", "lora_identity", "lora_medical"])
        lora_layout.addWidget(lora_label)
        lora_layout.addWidget(self.lora_combo)
        advanced_layout.addLayout(lora_layout)
        
        # 模型参数
        param_layout = QHBoxLayout()
        
        dim_layout = QVBoxLayout()
        dim_label = QLabel("维度(dim):")
        self.dim_spin = QSpinBox()
        self.dim_spin.setRange(128, 2048)
        self.dim_spin.setValue(512)
        dim_layout.addWidget(dim_label)
        dim_layout.addWidget(self.dim_spin)
        param_layout.addLayout(dim_layout)
        
        layers_layout = QVBoxLayout()
        layers_label = QLabel("层数(n_layers):")
        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 32)
        self.layers_spin.setValue(8)
        layers_layout.addWidget(layers_label)
        layers_layout.addWidget(self.layers_spin)
        param_layout.addLayout(layers_layout)
        
        seq_len_layout = QVBoxLayout()
        seq_len_label = QLabel("最大序列长度:")
        self.seq_len_spin = QSpinBox()
        self.seq_len_spin.setRange(512, 32768)
        self.seq_len_spin.setValue(8192)
        seq_len_layout.addWidget(seq_len_label)
        seq_len_layout.addWidget(self.seq_len_spin)
        param_layout.addLayout(seq_len_layout)
        
        advanced_layout.addLayout(param_layout)
        
        # MoE选项
        self.moe_check = QCheckBox("使用MoE (混合专家)")
        self.moe_check.setChecked(False)
        advanced_layout.addWidget(self.moe_check)
        
        # 生成参数
        gen_layout = QHBoxLayout()
        
        temp_layout = QVBoxLayout()
        temp_label = QLabel("温度(temperature):")
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 2.0)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(0.85)
        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.temp_spin)
        gen_layout.addLayout(temp_layout)
        
        top_p_layout = QVBoxLayout()
        top_p_label = QLabel("Top-p采样:")
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.1, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(0.85)
        top_p_layout.addWidget(top_p_label)
        top_p_layout.addWidget(self.top_p_spin)
        gen_layout.addLayout(top_p_layout)
        
        history_layout = QVBoxLayout()
        history_label = QLabel("历史对话条数:")
        self.history_spin = QSpinBox()
        self.history_spin.setRange(0, 20)
        self.history_spin.setValue(0)
        history_layout.addWidget(history_label)
        history_layout.addWidget(self.history_spin)
        gen_layout.addLayout(history_layout)
        
        advanced_layout.addLayout(gen_layout)
        
        config_layout.addWidget(self.advanced_frame)
        main_layout.addWidget(config_frame)
        
        # 开始按钮
        start_btn = QPushButton("开始聊天")
        start_btn.setFont(QFont("Arial", 14, QFont.Bold))
        start_btn.clicked.connect(self.start_chat)
        main_layout.addWidget(start_btn)
        
        # 样式表
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QFrame {
                background-color: white;
                border-radius: 10px;
                padding: 15px;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
            }
            QCheckBox {
                spacing: 5px;
            }
        """)
    
    def toggle_advanced(self):
        self.advanced_frame.setVisible(self.advanced_btn.isChecked())
        self.advanced_btn.setText("高级选项 ▲" if self.advanced_btn.isChecked() else "高级选项 ▼")
    
    def start_chat(self):
        try:
            print("正在收集参数...")
            # 收集参数
            args = argparse.Namespace()
            args.model_mode = self.model_mode_combo.currentIndex()
            args.load = self.load_mode_combo.currentIndex()
            args.lora_name = self.lora_combo.currentText()
            args.dim = self.dim_spin.value()
            args.n_layers = self.layers_spin.value()
            args.max_seq_len = self.seq_len_spin.value()
            args.use_moe = self.moe_check.isChecked()
            args.temperature = self.temp_spin.value()
            args.top_p = self.top_p_spin.value()
            args.history_cnt = self.history_spin.value()
            args.stream = True
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            args.out_dir = 'out'
            
            print("正在初始化tokenizer...")
            # 初始化tokenizer
            project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            tokenizer_path = project_root / "numlife" / "model" / "numlife_tokenizer"
            print(f"Tokenizer路径: {tokenizer_path}")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
                print("Tokenizer加载成功")
            except Exception as e:
                print(f"Tokenizer加载失败: {e}")
                QMessageBox.critical(self, "错误", f"加载tokenizer失败: {e}")
                return
            
            print("创建聊天窗口...")
            # 创建聊天窗口
            self.chat_window = ChatWindow(args, tokenizer)
            self.chat_window.show()
            self.close()
            print("聊天窗口已创建")
            
        except Exception as e:
            print(f"启动聊天出错: {e}")
            QMessageBox.critical(self, "错误", f"启动聊天失败: {e}")

class ChatWindow(QMainWindow):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.model = None
        self.messages = []
        
        self.setWindowTitle("numlife  聊天")
        self.setMinimumSize(800, 600)
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        # 主窗口
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 顶部状态栏
        status_bar = QFrame()
        status_bar.setFixedHeight(40)
        status_layout = QHBoxLayout()
        status_bar.setLayout(status_layout)
        
        self.status_label = QLabel("正在加载模型...")
        self.status_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        self.model_info = QLabel("")
        self.model_info.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.model_info)
        
        main_layout.addWidget(status_bar)
        
        # 聊天区域
        chat_area = QFrame()
        chat_layout = QVBoxLayout()
        chat_area.setLayout(chat_layout)
        
        # 消息显示区域
        self.message_area = QTextEdit()
        self.message_area.setReadOnly(True)
        self.message_area.setFont(QFont("Arial", 12))
        chat_layout.addWidget(self.message_area)
        
        # 输入区域
        input_frame = QFrame()
        input_layout = QHBoxLayout()
        input_frame.setLayout(input_layout)
        
        self.input_box = QTextEdit()
        self.input_box.setMaximumHeight(100)
        self.input_box.setFont(QFont("Arial", 12))
        self.input_box.setPlaceholderText("输入消息...")
        input_layout.addWidget(self.input_box)
        
        self.send_btn = QPushButton("发送")
        self.send_btn.setFixedWidth(80)
        self.send_btn.setFont(QFont("Arial", 12))
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)
        
        chat_layout.addWidget(input_frame)
        main_layout.addWidget(chat_area)
        
        # 样式表
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QFrame {
                background-color: white;
                border-radius: 5px;
                padding: 5px;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        
        # 初始状态
        self.send_btn.setEnabled(False)
    
    def load_model(self):
        self.loader = ModelLoader(self.args, self.tokenizer)
        self.loader.model_loaded.connect(self.on_model_loaded)
        self.loader.error_occurred.connect(self.on_model_error)
        self.loader.start()
    
    def on_model_loaded(self, model):
        self.model = model
        self.status_label.setText("模型加载完成")
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
        self.model_info.setText(f"参数: {param_count:.2f}M")
        self.send_btn.setEnabled(True)
        self.add_message("系统", "模型已加载完成，可以开始聊天！", is_system=True)
    
    def on_model_error(self, error_msg):
        self.status_label.setText(f"模型加载失败")
        self.add_message("系统", f"模型加载失败: {error_msg}", is_system=True)
        QMessageBox.critical(self, "错误", f"模型加载失败: {error_msg}")
    
    def send_message(self):
        message = self.input_box.toPlainText().strip()
        if not message:
            return
        
        self.input_box.clear()
        self.add_message("你", message, is_user=True)
        
        # 添加到消息历史
        self.messages = self.messages[-self.args.history_cnt:] if self.args.history_cnt else []
        self.messages.append({"role": "user", "content": message})
        
        # 在新线程中生成回复
        self.gen_thread = ModelLoader(self.args, self.tokenizer)
        self.gen_thread.model_loaded.connect(lambda model: self.generate_response(message, model))
        self.gen_thread.error_occurred.connect(lambda error: self.append_message("系统", f"生成回复时出错: {error}"))
        self.gen_thread.start()
        
        # 禁用发送按钮直到回复完成
        self.send_btn.setEnabled(False)
    
    def generate_response(self, prompt, model):
        try:
            with torch.no_grad():
                new_prompt = self.tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )[-self.args.max_seq_len + 1:] if self.args.model_mode != 0 else (self.tokenizer.bos_token + prompt)

                x = torch.tensor(self.tokenizer(new_prompt)['input_ids'], device=self.args.device).unsqueeze(0)
                outputs = model.generate(
                    x,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=self.args.max_seq_len,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    stream=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                # 重置当前回复框架
                self.current_response = None

                answer = ""
                for y in outputs:
                    current = self.tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                    if (current and current[-1] == '�') or not current:
                        continue
                    
                    new_text = current[len(answer):]
                    answer = current
                    self.append_message("AI", new_text, is_stream=True)
                
                # 保存到消息历史
                self.messages.append({"role": "assistant", "content": answer})

            # 启用发送按钮
            self.send_btn.setEnabled(True)

        except Exception as e:
            self.append_message("系统", f"生成回复时出错: {str(e)}", is_system=True)
            self.send_btn.setEnabled(True)

    def append_message(self, sender, text, is_stream=False, is_system=False):
        """追加或更新消息"""
        cursor = self.message_area.textCursor()
        
        if is_stream and self.current_response is not None:
            # 更新现有回复
            cursor.setPosition(self.current_response[0])
            cursor.setPosition(self.current_response[1], QTextCursor.KeepAnchor)
            format = cursor.charFormat()
            cursor.removeSelectedText()
            cursor.setCharFormat(format)
            cursor.insertText(self.current_response[2] + text)
            self.current_response = (self.current_response[0], cursor.position(), self.current_response[2] + text)
        else:
            # 添加新消息
            cursor.movePosition(QTextCursor.End)
            
            # 添加换行（如果不是第一条消息）
            if self.message_area.toPlainText():
                cursor.insertText("\n\n")
            
            # 设置发送者样式
            format = QTextCharFormat()
            if is_system:
                format.setForeground(QColor("#666666"))
            else:
                format.setForeground(QColor("#007AFF" if sender == "你" else "#4CAF50"))
            format.setFontWeight(QFont.Bold)
            cursor.setCharFormat(format)
            cursor.insertText(f"{sender}: ")
            
            # 设置消息文本样式
            format = QTextCharFormat()
            if is_system:
                format.setForeground(QColor("#666666"))
            else:
                format.setForeground(QColor("black"))
            cursor.setCharFormat(format)
            cursor.insertText(text)
            
            if is_stream:
                # 保存流式回复的位置信息
                self.current_response = (
                    cursor.position() - len(text),  # 开始位置
                    cursor.position(),              # 结束位置
                    text                           # 当前文本
                )
        
        # 滚动到底部
        self.message_area.verticalScrollBar().setValue(
            self.message_area.verticalScrollBar().maximum()
        )

    def add_message(self, sender, text, is_user=False, is_system=False):
        """添加新消息"""
        cursor = self.message_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # 添加换行（如果不是第一条消息）
        if self.message_area.toPlainText():
            cursor.insertText("\n\n")
        
        # 设置发送者样式
        format = QTextCharFormat()
        if is_system:
            format.setForeground(QColor("#666666"))
        else:
            format.setForeground(QColor("#007AFF" if is_user else "#4CAF50"))
        format.setFontWeight(QFont.Bold)
        cursor.setCharFormat(format)
        cursor.insertText(f"{sender}: ")
        
        # 设置消息文本样式
        format = QTextCharFormat()
        if is_system:
            format.setForeground(QColor("#666666"))
        else:
            format.setForeground(QColor("black"))
        cursor.setCharFormat(format)
        cursor.insertText(text)
        
        # 滚动到底部
        self.message_area.verticalScrollBar().setValue(
            self.message_area.verticalScrollBar().maximum()
        )

    def keyPressEvent(self, event):
        """处理按键事件"""
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.NoModifier:
            # 按下回车键发送消息
            self.send_message()
            event.accept()
        elif event.key() == Qt.Key_Return and event.modifiers() == Qt.ShiftModifier:
            # Shift+回车换行
            self.input_box.insertPlainText("\n")
            event.accept()
        else:
            super().keyPressEvent(event)

def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle("Fusion")
    
    # 创建并显示模型选择窗口
    window = ModelSelectionWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
