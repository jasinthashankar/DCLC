# Post-Quantum Cryptographic Protocols and Key Exchange

## 1. Module-Lattice Key Encapsulation (ML-KEM / Kyber-768)
Quantum-resistant key exchange establishes ephemeral shared secrets between communication endpoints using Module-Lattice-based Key Encapsulation Mechanisms (ML-KEM, standardized from Crystals-Kyber). Security parameters target NIST Level 3 equivalence, resilient against Shor algorithm attacks on future cryptanalytic quantum computers.

## 2. Module-Lattice Digital Signatures (ML-DSA / Dilithium)
Public key authentication and message integrity verification utilize Module-Lattice Digital Signature Algorithms (ML-DSA, standardized from Crystals-Dilithium). Public keys and signatures maintain bounded byte lengths suitable for inclusion in X.509 certificate chains and TLS 1.3 handshake extensions.

## 3. Hybrid Classical-Quantum TLS 1.3 Handshakes
During the cryptographic transition era, secure transport handshakes combine classical X25519 elliptic-curve Diffie-Hellman with post-quantum ML-KEM in a dual-encapsulation hybrid key exchange. Session encryption keys are derived through HKDF-Extract over the concatenated classical and post-quantum shared secrets.
