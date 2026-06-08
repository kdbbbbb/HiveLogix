# 各功能模块的 Blueprint 路由文件
# 新增功能示例：
#   from flask import Blueprint
#   dispatch_bp = Blueprint('dispatch', __name__)
#   @dispatch_bp.route('/status')
#   def status(): ...
# 然后在 backend/app.py 中注册：
#   from api.routes.dispatch import dispatch_bp
#   app.register_blueprint(dispatch_bp, url_prefix='/api/dispatch')
