# Distributed Storage Cluster Architecture Specification — Version 2.0

## 1. Overview and Cluster Topology
The distributed storage cluster provides a high-availability, fault-tolerant block and object store designed for enterprise multitenant environments. The cluster consists of three primary roles: metadata coordinators, storage worker daemons, and client ingress gateways. All nodes communicate over a dedicated low-latency backplane using synchronized clock domains and redundant top-of-rack switches.

## 2. Storage Engine and Log-Structured Merge Tree
The core persistence tier employs a log-structured merge (LSM) tree architecture optimized for high-throughput write workloads. Writes are sequentially appended to an in-memory active MemTable before flushing immutable SSTables to non-volatile flash media. Compaction workers operate in the background using leveled compaction policies to bound read amplification and reclaim invalidated storage space.

## 3. POSIX Compatibility and Inode Cache
File system emulation exposes a POSIX-compliant namespace to client applications via an in-kernel VFS module. Inode lookups are accelerated through a lock-free concurrent hash table cache with LRU eviction. Directory entries maintain monotonic sequence numbers to guarantee atomic renames and consistent readdir traversals across distributed mount points.

## 4. Disk Block Layout and Extent Allocation
Physical storage devices are formatted with a uniform 4KB block size grouped into 64MB contiguous allocation extents. The extent allocator uses buddy allocation bitmaps to prevent internal fragmentation and guarantee deterministic sequential read performance. Free space bitmaps are journaled periodically to prevent allocation leaks following sudden power loss.

## 5. Worker Node Memory Allocation
Worker nodes are strictly provisioned with 128GB ECC DDR5 memory per physical node. The memory manager designates 64GB specifically for the read block cache, 32GB for active MemTable allocations, 16GB for network socket ring buffers, and reserves 16GB for the operating system kernel and management agents. Out-of-memory killer daemon thresholds trigger proactive cache shedding at 95% allocation.

## 6. Network Ingress Gateway
The ingress gateway service listens on TCP port 9443 with mandatory TLS 1.3 encryption across all client and administrative communication. Cleartext HTTP/1.1 transport is completely prohibited and rejected at the socket handshake level. Maximum concurrent socket connections per gateway daemon are expanded to 16384 using epoll event loops.

## 7. Data-at-Rest Encryption
Data-at-rest encryption is upgraded to ChaCha20-Poly1305 stream cipher suites to provide uniform cryptographic performance across heterogeneous ARM64 and x86-64 storage servers without requiring specialized AES-NI silicon coprocessors. Ephemeral volume keys are rotated every 90 days and anchored into an internal decentralized vault KMS.

## 8. Raft Consensus and Failure Domains
Metadata replication achieves linearizable consistency using the Raft consensus algorithm. The election timeout interval is configured with a base period of 5000ms and a heartbeat tick of 1000ms. Consensus quorums require a strict majority across at least three distinct availability zones to survive simultaneous rack and power strip outages.

## 9. eBPF Kernel Telemetry and Socket Tracing
Worker nodes deploy an in-kernel eBPF agent attached to kprobes and tracepoints along the block I/O submission path. The eBPF collector records microsecond-level NVMe request queue latencies, socket buffer drops, and TCP retransmissions, exporting Prometheus-compatible metrics via shared ring buffers without context switch penalties.

## 10. NVMe-over-Fabrics Target Configuration
Direct block storage attachment is implemented via the NVMe-over-Fabrics (NVMe-oF) protocol over 100GbE RoCEv2 (RDMA over Converged Ethernet). Client initiators establish zero-copy kernel-bypass connections directly to remote storage controller namespaces, reducing tail p99 I/O latency to under 45 microseconds.

## 11. Dynamic Adaptive Rate Limiting
Client requests are throttled at the ingress gateway using a dynamic token bucket algorithm informed by real-time storage backpressure. The rate limiter dynamically computes tenant I/O quotas based on fair-share credit allocations, ensuring bursty analytics jobs do not degrade latency-critical transactional workloads.

## 12. Chunk Garbage Collection
Unreferenced data chunks resulting from file truncations and snapshot deletions are collected via a background generational mark-and-sweep garbage collector. Reclamation runs during low-traffic maintenance windows between 02:00 and 04:00 UTC. Chunks are safe for deletion only after their reference counter reaches zero across all active snapshot manifests.

## 13. Write-Ahead Logging and Crash Recovery
Every modifying transaction appends a synchronous entry to the write-ahead log (WAL) before updating in-memory state. WAL segments are pre-allocated in 64MB extents and flushed to disk with explicit `fsync` barriers. Crash recovery replays the WAL from the most recent checkpoint, reconstructing consistent memory state within under 15 seconds.
