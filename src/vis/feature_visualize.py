import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import matplotlib
import os
import numpy as np
from nuplan.common.actor_state.state_representation import Point2D

vis_save_path = "/home/wang/Project/nuplan/planTF/exp/visualizations"
os.makedirs(vis_save_path, exist_ok=True)
vis_counter = 0


def visualize_agent_features(
    position: np.ndarray,
    velocity: np.ndarray,
    shape: np.ndarray,
    category: np.ndarray,
    valid_mask: np.ndarray,
    present_idx: int,
    query_xy: Point2D,
    radius: float
):
    """可视化智能体特征"""
    global vis_counter
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 颜色定义
    agent_colors = {
        0: '#F44336',  # EGO - 红色
        1: '#2196F3',  # VEHICLE - 蓝色
        2: '#4CAF50',  # PEDESTRIAN - 绿色
        3: '#FF9800'   # BICYCLE - 橙色
    }
    
    agent_names = {
        0: 'EGO',
        1: 'VEHICLE',
        2: 'PEDESTRIAN',
        3: 'BICYCLE'
    }

    N, T = position.shape[:2]
    
    # 1. 绘制当前时刻的智能体位置和形状
    ax1 = axes[0, 0]
    ax1.set_title(f'Current Agent Distribution (N={N})', fontsize=14)
    
    for i in range(N):
        if valid_mask[i, present_idx]:
            pos = position[i, present_idx]
            cat = category[i]
            color = agent_colors.get(cat, '#9E9E9E')
            
            # 绘制智能体边界框
            width, length = shape[i, present_idx]
            rect = patches.Rectangle(
                (pos[0] - length/2, pos[1] - width/2),
                length, width,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.6
            )
            ax1.add_patch(rect)
            
            # 标注智能体ID
            ax1.text(pos[0], pos[1], str(i), ha='center', va='center', 
                    color='white', fontweight='bold', fontsize=8)
    
    # 标记查询中心点
    ax1.scatter(query_xy.x, query_xy.y, color='black', s=200, marker='x', 
                linewidth=3, label='Query Center')
    
    ax1.set_xlim(query_xy.x - radius, query_xy.x + radius)
    ax1.set_ylim(query_xy.y - radius, query_xy.y + radius)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. 绘制智能体轨迹
    ax2 = axes[0, 1]
    ax2.set_title('Agent History and Future Trajectories', fontsize=14)
    
    for i in range(N):
        cat = category[i]
        color = agent_colors.get(cat, '#9E9E9E')
        
        # 绘制轨迹线
        traj_points = []
        for t in range(T):
            if valid_mask[i, t]:
                traj_points.append(position[i, t])
        
        if len(traj_points) > 1:
            traj_points = np.array(traj_points)
            ax2.plot(traj_points[:, 0], traj_points[:, 1], 
                    color=color, alpha=0.7, linewidth=2)
            
            # 标记起点和终点
            ax2.scatter(traj_points[0, 0], traj_points[0, 1], 
                        color=color, s=50, marker='o', alpha=0.8)
            ax2.scatter(traj_points[-1, 0], traj_points[-1, 1], 
                        color=color, s=50, marker='s', alpha=0.8)
        
        # 特别标记当前时刻位置
        if valid_mask[i, present_idx]:
            pos = position[i, present_idx]
            ax2.scatter(pos[0], pos[1], color=color, s=100, 
                        marker='*', edgecolor='black', linewidth=1)
    
    ax2.scatter(query_xy.x, query_xy.y, color='black', s=200, marker='x', linewidth=3)
    ax2.set_xlim(query_xy.x - radius, query_xy.x + radius)
    ax2.set_ylim(query_xy.y - radius, query_xy.y + radius)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # 3. 绘制速度向量
    ax3 = axes[1, 0]
    ax3.set_title('Agent Velocity Vectors', fontsize=14)
    
    for i in range(N):
        if valid_mask[i, present_idx]:
            pos = position[i, present_idx]
            vel = velocity[i, present_idx]
            cat = category[i]
            color = agent_colors.get(cat, '#9E9E9E')
            
            # 绘制速度向量
            speed = np.linalg.norm(vel)
            if speed > 0.1:  # 只显示有明显速度的智能体
                arrow_scale = min(10.0, 20.0 / max(speed, 1.0))  # 自适应箭头长度
                ax3.arrow(pos[0], pos[1], vel[0] * arrow_scale, vel[1] * arrow_scale,
                            head_width=2, head_length=1.5, fc=color, ec=color, alpha=0.8)
                
                # 显示速度值
                ax3.text(pos[0] + vel[0] * arrow_scale, pos[1] + vel[1] * arrow_scale,
                        f'{speed:.1f}m/s', ha='center', va='bottom', fontsize=8)
            
            # 绘制智能体位置
            ax3.scatter(pos[0], pos[1], color=color, s=80, alpha=0.6)
    
    ax3.scatter(query_xy.x, query_xy.y, color='black', s=200, marker='x', linewidth=3)
    ax3.set_xlim(query_xy.x - radius, query_xy.x + radius)
    ax3.set_ylim(query_xy.y - radius, query_xy.y + radius)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # 4. 智能体类型统计
    ax4 = axes[1, 1]
    ax4.set_title('Agent Type Distribution', fontsize=14)
    
    # 统计各类型智能体数量
    valid_agents = valid_mask[:, present_idx]
    type_counts = {}
    for i in range(N):
        if valid_agents[i]:
            cat = category[i]
            type_name = agent_names.get(cat, f'Unknown({cat})')
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    if type_counts:
        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors_list = [agent_colors.get(i, '#9E9E9E') 
                        for i, name in enumerate(agent_names.values()) 
                        if name in types]
        
        bars = ax4.bar(types, counts, color=colors_list, alpha=0.7)
        
        # 在柱状图上显示数值
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylabel('Count')
        ax4.set_ylim(0, max(counts) * 1.2)
    else:
        ax4.text(0.5, 0.5, 'No valid agents', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14)
    
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(vis_save_path, f'agent_features_{vis_counter:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    vis_counter += 1


def visualize_map_features(
    point_position: np.ndarray,
    point_vector: np.ndarray, 
    polygon_center: np.ndarray,
    polygon_type: np.ndarray,
    polygon_on_route: np.ndarray,
    polygon_tl_status: np.ndarray,
    query_xy: Point2D,
    radius: float
):
    """绘制地图特征可视化图"""
    global vis_counter
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # 颜色定义
    colors = {
        'lane': '#4CAF50',           # 绿色 - 车道
        'lane_connector': '#FF9800', # 橙色 - 车道连接器
        'crosswalk': '#9C27B0',      # 紫色 - 人行横道
        'on_route': '#2196F3',       # 蓝色 - 在路径上
        'ego': '#F44336'             # 红色 - 自车位置
    }
    
    # 交通灯状态颜色
    tl_colors = {
        0: '#9E9E9E',  # UNKNOWN - 灰色
        1: '#4CAF50',  # GREEN - 绿色
        2: '#FFEB3B',  # YELLOW - 黄色
        3: '#F44336'   # RED - 红色
    }

    # 1. 绘制所有采样点
    ax1 = axes[0, 0]
    ax1.set_title(f'Map Sample Points (M={len(point_position)}, P={point_position.shape[2]})', fontsize=14)
    
    for i in range(len(point_position)):
        # 根据道路类型选择颜色
        type_idx = polygon_type[i]
        if type_idx == 0:  # LANE
            color = colors['lane']
            label = 'Lane'
        elif type_idx == 1:  # LANE_CONNECTOR
            color = colors['lane_connector'] 
            label = 'Lane Connector'
        else:  # CROSSWALK
            color = colors['crosswalk']
            label = 'Crosswalk'
        
        # 绘制三条边界线
        for side in range(3):  # 中心线, 左边界, 右边界
            points = point_position[i, side]
            ax1.plot(points[:, 0], points[:, 1], color=color, alpha=0.7, linewidth=1)
            
            # 绘制采样点
            ax1.scatter(points[:, 0], points[:, 1], color=color, s=8, alpha=0.8)
    
    # 标记自车位置
    ax1.scatter(query_xy.x, query_xy.y, color=colors['ego'], s=200, marker='*', 
                label='Ego Position', zorder=10)
    
    # 绘制查询半径
    circle = plt.Circle((query_xy.x, query_xy.y), radius, fill=False, 
                       color='black', linestyle='--', alpha=0.5)
    ax1.add_patch(circle)
    
    ax1.set_xlim(query_xy.x - radius, query_xy.x + radius)
    ax1.set_ylim(query_xy.y - radius, query_xy.y + radius)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. 绘制方向向量
    ax2 = axes[0, 1]
    ax2.set_title('Direction Vectors Visualization', fontsize=14)
    
    for i in range(len(point_position)):
        type_idx = polygon_type[i]
        if type_idx == 0:
            color = colors['lane']
        elif type_idx == 1:
            color = colors['lane_connector']
        else:
            color = colors['crosswalk']
        
        # 只绘制中心线的方向向量
        points = point_position[i, 0]  # 中心线
        vectors = point_vector[i, 0]   # 中心线方向向量
        
        # 每隔几个点绘制一个箭头
        for j in range(0, len(points), 3):
            if j < len(vectors):
                ax2.arrow(points[j, 0], points[j, 1], 
                         vectors[j, 0] * 5, vectors[j, 1] * 5,
                         head_width=2, head_length=1, fc=color, ec=color, alpha=0.7)
    
    ax2.scatter(query_xy.x, query_xy.y, color=colors['ego'], s=200, marker='*')
    ax2.set_xlim(query_xy.x - radius, query_xy.x + radius)
    ax2.set_ylim(query_xy.y - radius, query_xy.y + radius)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # 3. 绘制路径规划相关信息
    ax3 = axes[1, 0]
    ax3.set_title('Route Planning Information', fontsize=14)
    
    for i in range(len(polygon_center)):
        center = polygon_center[i, :2]
        
        if polygon_on_route[i]:
            color = colors['on_route']
            marker = 's'  # 方形
            size = 100
            label = 'On Planned Route'
        else:
            type_idx = polygon_type[i]
            if type_idx == 0:
                color = colors['lane']
            elif type_idx == 1:
                color = colors['lane_connector']
            else:
                color = colors['crosswalk']
            marker = 'o'  # 圆形
            size = 50
            label = 'Not On Route'
        
        ax3.scatter(center[0], center[1], color=color, s=size, marker=marker, alpha=0.8)
    
    ax3.scatter(query_xy.x, query_xy.y, color=colors['ego'], s=200, marker='*')
    ax3.set_xlim(query_xy.x - radius, query_xy.x + radius)
    ax3.set_ylim(query_xy.y - radius, query_xy.y + radius)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # 4. 绘制交通灯状态
    ax4 = axes[1, 1]
    ax4.set_title('Traffic Light Status', fontsize=14)
    
    for i in range(len(polygon_center)):
        center = polygon_center[i, :2]
        tl_status = polygon_tl_status[i]
        
        color = tl_colors.get(tl_status, '#9E9E9E')
        
        if tl_status == 0:  # UNKNOWN
            marker = 'o'
            size = 30
        else:
            marker = 's'
            size = 80
        
        ax4.scatter(center[0], center[1], color=color, s=size, marker=marker, alpha=0.8)
    
    ax4.scatter(query_xy.x, query_xy.y, color=colors['ego'], s=200, marker='*')
    ax4.set_xlim(query_xy.x - radius, query_xy.x + radius)
    ax4.set_ylim(query_xy.y - radius, query_xy.y + radius)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tl_colors[0], 
                  markersize=8, label='Unknown'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=tl_colors[1], 
                  markersize=8, label='Green'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=tl_colors[2], 
                  markersize=8, label='Yellow'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=tl_colors[3], 
                  markersize=8, label='Red'),
    ]
    ax4.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(vis_save_path, f'map_features_{vis_counter:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    vis_counter += 1
