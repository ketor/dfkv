# dfkv 可观测性（Metrics / 集群视图）

> v1.4.0 起。所有指标均为**新增**、对数据面零性能影响（热路径仅 relaxed 原子；延迟为 1/64 采样；HTTP 在独立端口/线程）。默认不开端口 → 行为与旧版一致。

## 1. 抓取端点

| 进程 | 开关 | 端点 |
|---|---|---|
| `dfkv_server` | `--metrics-port <p>`（缺省=关） | `GET /metrics`、`GET /healthz` |
| `dfkv_mds` | `--metrics-port <p>`（缺省=关） | `GET /metrics`、`GET /healthz` |

身份标签：`dfkv_server --id <node> --group <g>` → 所有序列带 `{node,group}`；不设则**无标签**（向后兼容）。Prometheus 抓取自带 `instance`/`job`。

```bash
dfkv_server --dir /mnt/d1,/mnt/d2 --port 12000 --rdma-port 12001 \
            --id gpu1-0001 --group glm --mds 10.0.0.1:9400 --advertise 10.0.0.11:12000 \
            --metrics-port 9100
curl -s 127.0.0.1:9100/metrics
```

## 2. 集群 / 环视图（CLI）

```bash
dfkvctl ring     --mds <ep,...> --group <g>   # 成员表 + 一致性哈希环 vnode 分布/占比
dfkvctl stat --all --mds <ep,...> --group <g> # 逐节点指标 + 集群聚合（容量/对象/命中率）
dfkvctl stat <ip:port>                        # 单节点原始 /metrics 文本（旧用法不变）
```

## 3. 指标清单

### 3.1 cache 节点（`dfkv_server` /metrics）
| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_build_info{version,transport}` | gauge | 版本 + 构建传输（rdma/tcp），恒为 1 |
| `dfkv_uptime_seconds` | gauge | 启动至今秒数 |
| `dfkv_cache_put_total` / `cache_hit_total` / `cache_miss_total` | counter | PUT / GET 命中 / GET 未命中 |
| `dfkv_exist_hit_total` / `exist_miss_total` | counter | Exist 命中 / 未命中 |
| `dfkv_bytes_written_total` / `bytes_read_total` | counter | 读写字节 |
| `dfkv_accepts_total` | counter | 累计 TCP accept |
| `dfkv_open_connections` | gauge | 当前打开连接数 |
| `dfkv_evictions_total` / `evicted_bytes_total` | counter | 淘汰对象数 / 字节 |
| `dfkv_errors_total{op,status}` | counter | 失败 op（put/get io、invalid） |
| `dfkv_objects` / `used_bytes` / `disks` | gauge | 对象数 / 占用 / 盘数 |
| `dfkv_disk_used_bytes{disk}` / `dfkv_disk_objects{disk}` | gauge | 每盘占用 / 对象 |
| `dfkv_op_latency_seconds{op}` | histogram | **1/64 采样**的 get/put 服务端延迟（50µs–100ms 桶） |

RDMA 构建额外（折叠进同一 /metrics）：
| `dfkv_rdma_completions_total` / `completion_errors_total` | counter | RDMA 请求完成 / 错误完成 |
| `dfkv_rdma_active_conns` | gauge | 当前服务中的 RDMA 连接 |
| `dfkv_rdma_idle_reclaims_total` | counter | 空闲超时回收的连接数 |

### 3.2 MDS（`dfkv_mds` /metrics）
| 指标 | 类型 | 含义 |
|---|---|---|
| `dfkv_mds_register_requests_total` / `keepalives_total` | counter | 注册 / 心跳 |
| `dfkv_mds_list_requests_total` | counter | ListMembers 次数 |
| `dfkv_mds_lease_grants_total` | counter | etcd lease 授予 |
| `dfkv_mds_etcd_errors_total` | counter | etcd I/O 失败 |
| `dfkv_mds_members` | gauge | 上次 List 返回的成员数 |

### 3.3 客户端（SGLang 插件 /metrics，经 prometheus_client）
插件后台轮询线程读 C 客户端快照（`client_stats_poll_s`，默认 10s，0=关）并镜像为带 `{tp_rank}` 的 Counter：
| 指标 | 含义 |
|---|---|
| `dfkv_client_ops_served_total` | 收到响应的 op 数 |
| `dfkv_client_io_errors_total` | 客户端观察到的传输失败 |
| `dfkv_client_unhealthy_skips_total` | 因 peer 熔断短路的 op |
| `dfkv_client_peer_marked_bad_total` / `peer_recovered_total` | peer 熔断 / 恢复切换 |

插件直接暴露（per-batch）：
| `dfkv_client_set_calls/pages/ok_pages/bytes_total{tp_rank}` | set 量 |
| `dfkv_client_get_calls/pages/hit_pages/bytes_total{tp_rank}` | get 量 |
| `dfkv_client_set_seconds{tp_rank}` / `get_seconds{tp_rank}` | batch 调用耗时直方图 |

C 客户端快照里还含传输级（RDMA 构建）：`dfkv_rdma_client_conns_opened_total`、`mr_regions`、`rail_conns_total{dev}`（NUMA 选轨分布）。

## 4. 性能保证

- 热路径计数：`std::atomic` relaxed `fetch_add`，零锁零分配（与既有模式一致）。
- 延迟：1/64 采样（`Sampler` 掩码），仅采样到的 op 读 vDSO 时钟。
- HTTP `/metrics`：独立端口 + 线程，仅 scrape 时把原子读成文本。
- 客户端计数搭在 `PeerHealth` 既有锁段上（无新增锁）；per-peer 仅错误路径（罕见）。
- 插件轮询线程睡眠态、按间隔触发，不在 per-batch/per-page 路径。
- 验证：每 PR 跑 RDMA Soft-RoCE loopback + ThreadSanitizer 全绿。
