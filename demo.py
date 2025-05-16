from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QTextEdit, QPushButton, QListWidget, 
                            QListWidgetItem, QFrame)
from PyQt5.QtCore import Qt, QSize, QMargins
from PyQt5.QtGui import QFont, QColor, QPixmap, QIcon

class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("聊天应用")
        self.setMinimumSize(800, 600)
        self.initUI()
        
    def initUI(self):
        # 主布局 - 水平分割
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 左侧联系人列表 (占1/4宽度)
        self.contact_list = self.create_contact_list()
        main_layout.addWidget(self.contact_list, stretch=1)
        
        # 右侧聊天区域 (占3/4宽度)
        self.chat_area = self.create_chat_area()
        main_layout.addWidget(self.chat_area, stretch=3)
        
        self.setLayout(main_layout)
        
        # 添加一些测试数据
        self.add_test_contacts()
        self.add_test_messages()
    
    def create_contact_list(self):
        """创建左侧联系人列表"""
        contact_frame = QFrame()
        contact_frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border-right: 1px solid #e6e6e6;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 搜索框
        search_bar = QTextEdit()
        search_bar.setPlaceholderText("搜索")
        search_bar.setMaximumHeight(40)
        search_bar.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 10px;
                margin: 10px;
            }
        """)
        layout.addWidget(search_bar)
        
        # 联系人列表
        self.contact_list_widget = QListWidget()
        self.contact_list_widget.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
            }
            QListWidget::item {
                border-bottom: 1px solid #e6e6e6;
                padding: 10px;
            }
            QListWidget::item:hover {
                background-color: #e6e6e6;
            }
            QListWidget::item:selected {
                background-color: #d9d9d9;
            }
        """)
        layout.addWidget(self.contact_list_widget)
        
        contact_frame.setLayout(layout)
        return contact_frame
    
    def create_chat_area(self):
        """创建右侧聊天区域"""
        chat_frame = QFrame()
        chat_frame.setStyleSheet("""
            QFrame {
                background-color: #e5ddd5;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 聊天标题栏
        title_bar = QFrame()
        title_bar.setMaximumHeight(60)
        title_bar.setStyleSheet("""
            QFrame {
                background-color: #ededed;
                border-bottom: 1px solid #d1d1d1;
            }
        """)
        
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(15, 0, 15, 0)
        
        # 联系人头像
        avatar = QLabel()
        avatar.setPixmap(QPixmap(":user.png").scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        title_layout.addWidget(avatar)
        
        # 联系人名称
        self.contact_name = QLabel("未选择联系人")
        self.contact_name.setFont(QFont("Arial", 12, QFont.Bold))
        title_layout.addWidget(self.contact_name, stretch=1)
        
        # 功能按钮
        btn_call = QPushButton()
        btn_call.setIcon(QIcon(":call.png"))
        btn_call.setIconSize(QSize(20, 20))
        btn_call.setFlat(True)
        title_layout.addWidget(btn_call)
        
        btn_video = QPushButton()
        btn_video.setIcon(QIcon(":video.png"))
        btn_video.setIconSize(QSize(20, 20))
        btn_video.setFlat(True)
        title_layout.addWidget(btn_video)
        
        btn_more = QPushButton()
        btn_more.setIcon(QIcon(":more.png"))
        btn_more.setIconSize(QSize(20, 20))
        btn_more.setFlat(True)
        title_layout.addWidget(btn_more)
        
        title_bar.setLayout(title_layout)
        layout.addWidget(title_bar)
        
        # 消息显示区域
        self.message_area = QListWidget()
        self.message_area.setStyleSheet("""
            QListWidget {
                background-color: #e5ddd5;
                border: none;
            }
            QListWidget::item {
                border: none;
                padding: 5px;
            }
        """)
        layout.addWidget(self.message_area, stretch=1)
        
        # 消息输入区域
        input_frame = QFrame()
        input_frame.setMaximumHeight(120)
        input_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border-top: 1px solid #d1d1d1;
            }
        """)
        
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(10, 10, 10, 10)
        
        # 输入框
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("输入消息...")
        self.input_box.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #d1d1d1;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        input_layout.addWidget(self.input_box, stretch=1)
        
        # 发送按钮和功能按钮
        btn_layout = QHBoxLayout()
        
        btn_emoji = QPushButton("😊")
        btn_emoji.setFlat(True)
        btn_layout.addWidget(btn_emoji)
        
        btn_attach = QPushButton("📎")
        btn_attach.setFlat(True)
        btn_layout.addWidget(btn_attach)
        
        btn_layout.addStretch(1)
        
        btn_send = QPushButton("发送")
        btn_send.setStyleSheet("""
            QPushButton {
                background-color: #07C160;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #05A14E;
            }
        """)
        btn_send.clicked.connect(self.send_message)
        btn_layout.addWidget(btn_send)
        
        input_layout.addLayout(btn_layout)
        input_frame.setLayout(input_layout)
        layout.addWidget(input_frame)
        
        chat_frame.setLayout(layout)
        return chat_frame
    
    def add_test_contacts(self):
        """添加测试联系人"""
        contacts = [
            {"name": "张三", "avatar": ":user1.png", "last_msg": "你好！", "time": "12:30"},
            {"name": "李四", "avatar": ":user2.png", "last_msg": "在吗？", "time": "10:15"},
            {"name": "王五", "avatar": ":user3.png", "last_msg": "晚上一起吃饭", "time": "昨天"},
            {"name": "赵六", "avatar": ":user4.png", "last_msg": "文件已发送", "time": "星期一"},
            {"name": "微信群", "avatar": ":group.png", "last_msg": "Alice: 大家周末愉快", "time": "星期日"},
        ]
        
        for contact in contacts:
            item = QListWidgetItem()
            item.setSizeHint(QSize(200, 60))
            
            widget = QWidget()
            layout = QHBoxLayout()
            layout.setContentsMargins(10, 5, 10, 5)
            
            # 头像
            avatar = QLabel()
            avatar.setPixmap(QPixmap(contact["avatar"]).scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            layout.addWidget(avatar)
            
            # 联系人信息
            info_layout = QVBoxLayout()
            info_layout.setSpacing(2)
            
            name_layout = QHBoxLayout()
            name_label = QLabel(contact["name"])
            name_label.setFont(QFont("Arial", 10, QFont.Bold))
            name_layout.addWidget(name_label, stretch=1)
            
            time_label = QLabel(contact["time"])
            time_label.setFont(QFont("Arial", 8))
            time_label.setStyleSheet("color: #888888;")
            name_layout.addWidget(time_label)
            
            info_layout.addLayout(name_layout)
            
            msg_label = QLabel(contact["last_msg"])
            msg_label.setFont(QFont("Arial", 9))
            msg_label.setStyleSheet("color: #888888;")
            info_layout.addWidget(msg_label)
            
            layout.addLayout(info_layout, stretch=1)
            widget.setLayout(layout)
            
            self.contact_list_widget.addItem(item)
            self.contact_list_widget.setItemWidget(item, widget)
    
    def add_test_messages(self):
        """添加测试消息"""
        messages = [
            {"sender": "other", "text": "你好！最近怎么样？", "time": "12:30"},
            {"sender": "me", "text": "我很好，谢谢关心！你呢？", "time": "12:32"},
            {"sender": "other", "text": "我也不错，最近在忙一个新项目", "time": "12:33"},
            {"sender": "other", "text": "有空一起喝咖啡聊聊吗？", "time": "12:34"},
            {"sender": "me", "text": "好啊，周五下午怎么样？", "time": "12:35"},
        ]
        
        for msg in messages:
            self.add_message(msg["sender"], msg["text"], msg["time"])
    
    def add_message(self, sender, text, time):
        """添加一条消息到聊天区域"""
        item = QListWidgetItem()
        item.setSizeHint(QSize(self.message_area.width(), 60))
        
        widget = QWidget()
        layout = QHBoxLayout()
        
        if sender == "me":
            # 自己发送的消息 - 右对齐
            layout.addStretch(1)
            
            msg_frame = QFrame()
            msg_frame.setStyleSheet("""
                QFrame {
                    background-color: #95EC69;
                    border-radius: 5px;
                    padding: 8px;
                    max-width: 300px;
                }
            """)
            
            msg_layout = QVBoxLayout()
            msg_layout.setContentsMargins(5, 5, 5, 5)
            
            msg_text = QLabel(text)
            msg_text.setWordWrap(True)
            msg_text.setFont(QFont("Arial", 10))
            msg_layout.addWidget(msg_text)
            
            time_label = QLabel(time)
            time_label.setAlignment(Qt.AlignRight)
            time_label.setFont(QFont("Arial", 8))
            time_label.setStyleSheet("color: #666666;")
            msg_layout.addWidget(time_label)
            
            msg_frame.setLayout(msg_layout)
            layout.addWidget(msg_frame)
        else:
            # 对方发送的消息 - 左对齐
            # 头像
            avatar = QLabel()
            avatar.setPixmap(QPixmap(":user1.png").scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            layout.addWidget(avatar)
            
            msg_frame = QFrame()
            msg_frame.setStyleSheet("""
                QFrame {
                    background-color: white;
                    border-radius: 5px;
                    padding: 8px;
                    max-width: 300px;
                }
            """)
            
            msg_layout = QVBoxLayout()
            msg_layout.setContentsMargins(5, 5, 5, 5)
            
            msg_text = QLabel(text)
            msg_text.setWordWrap(True)
            msg_text.setFont(QFont("Arial", 10))
            msg_layout.addWidget(msg_text)
            
            time_label = QLabel(time)
            time_label.setAlignment(Qt.AlignRight)
            time_label.setFont(QFont("Arial", 8))
            time_label.setStyleSheet("color: #666666;")
            msg_layout.addWidget(time_label)
            
            msg_frame.setLayout(msg_layout)
            layout.addWidget(msg_frame)
            
            layout.addStretch(1)
        
        widget.setLayout(layout)
        self.message_area.addItem(item)
        self.message_area.setItemWidget(item, widget)
        self.message_area.scrollToBottom()
    
    def send_message(self):
        """发送消息"""
        text = self.input_box.toPlainText().strip()
        if text:
            self.add_message("me", text, "刚刚")
            self.input_box.clear()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    
    # 设置应用程序图标和样式
    app.setStyle("Fusion")
    
    chat_app = ChatApp()
    chat_app.show()
    sys.exit(app.exec_())
