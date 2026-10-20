# Distributed Storage Cluster Architecture Specification — Version 1.0

## 1. Overview and Cluster Topology
The distributed storage cluster provides a high-availability, fault-tolerant block and object store designed for enterprise multitenant environments. The cluster consists of three primary roles: metadata coordinators, storage worker daemons, and client ingress gateways. All nodes communicate over a dedicated low-latency backplane using synchronized clock domains and redundant top-of-rack switches.

## 2. Storage Engine and Log-Structured Merge Tree
The core persistence tier employs a log-structured merge (LSM) tree architecture optimized for high-throughput write workloads. Writes are sequentially appended to an in-memory active MemTable before flushing immutable SSTables to non-volatile flash media. Compaction workers operate in the background using leveled compaction policies to bound read amplification and reclaim invalidated storage space.

## 3. POSIX Compatibility and Inode Cache
File system emulation exposes a POSIX-compliant namespace to client applications via an in-kernel VFS module. Inode lookups are accelerated through a lock-free concurrent hash table cache with LRU eviction. Directory entries maintain monotonic sequence numbers to guarantee atomic renames and consistent readdir traversals across distributed mount points.

## 4. Disk Block Layout and Extent Allocation
Physical storage devices are formatted with a uniform 4KB block size grouped into 64MB contiguous allocation extents. The extent allocator uses buddy allocation bitmaps to prevent internal fragmentation and guarantee deterministic sequential read performance. Free space bitmaps are journaled periodically to prevent allocation leaks following sudden power loss.

## 5. Worker Node Memory Allocation
Worker nodes are strictly provisioned with 32GB ECC DDR4 memory per physical node. The memory manager designates 16GB specifically for the read block cache, 8GB for active MemTable allocations, 4GB for network socket ring buffers, and reserves 4GB for the operating system kernel and management agents. Out-of-memory killer daemon thresholds trigger proactive cache shedding at 90% allocation.

## 6. Network Ingress Gateway
The ingress gateway service listens on TCP port 8080 for standard administrative REST traffic and data client requests. Unencrypted HTTP/1.1 transport is permitted within the secure management perimeter, with optional TLS termination delegated to an external reverse proxy. Maximum concurrent socket connections per gateway daemon are capped at 4096.

## 7. Data-at-Rest Encryption
Data-at-rest encryption is enforced globally across all persistent media using AES-256-GCM authenticated cipher blocks. Each storage volume maintains an encrypted Master Key wrapped by an external HashiCorp Vault Key Management Service (KMS). Hardware acceleration is leveraged through x86-64 AES-NI instruction sets to ensure zero throughput degradation.

## 8. Raft Consensus and Failure Domains
Metadata replication achieves linearizable consistency using the Raft consensus algorithm. The election timeout interval is configured with a base period of 5000ms and a heartbeat tick of 1000ms. Consensus quorums require a strict majority across at least three distinct availability zones to survive simultaneous rack and power strip outages.

## 9. Legacy XML Configuration Parser
Cluster configuration files are serialized in XML 1.0 syntax conforming to the cluster-spec-v1 schema definition. Configuration files are validated during daemon startup using an XML DOM parser. Deprecated attributes including `legacy-failover-mode` and `xml-rpc-target` are parsed for backward compatibility with legacy management scripts.

## 10. Custom Binary RPC Protocol v1
Inter-node remote procedure calls utilize the legacy binary RPC protocol v1. Packets consist of a fixed 16-byte header with a 32-bit magic preamble, message type integer, and payload length. Serialization relies on custom endianness byte packing without schema evolution or dynamic field reflection.

## 11. Chunk Garbage Collection
Unreferenced data chunks resulting from file truncations and snapshot deletions are collected via a background generational mark-and-sweep garbage collector. Reclamation runs during low-traffic maintenance windows between 02:00 and 04:00 UTC. Chunks are safe for deletion only after their reference counter reaches zero across all active snapshot manifests.

## 12. Write-Ahead Logging and Crash Recovery
Every modifying transaction appends a synchronous entry to the write-ahead log (WAL) before updating in-memory state. WAL segments are pre-allocated in 64MB extents and flushed to disk with explicit `fsync` barriers. Crash recovery replays the WAL from the most recent checkpoint, reconstructing consistent memory state within under 15 seconds.
