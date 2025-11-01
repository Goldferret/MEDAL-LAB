# DOFBOT Pro - Archived Hardware-Level Implementation

This directory contains the original hardware-level implementation of the DOFBOT Pro robot node. It has been archived in favor of the ROS-based implementation but is preserved for reference and diagnostic purposes.

## What is This?

This is the **legacy implementation** that provided direct hardware control of the DOFBOT Pro robotic arm using:
- Direct I2C servo communication via `Arm_Lib.py`
- Hardware-level camera integration with `pyorbbecsdk`
- Component-based modular architecture
- MADSci-compliant REST API

## Why Was It Archived?

The system has transitioned to a **ROS-based implementation** (`../dofbot-pro-ros/`) because:
- Better integration with manufacturer's MoveIT motion planning
- Improved development workflow with ROS ecosystem
- Easier maintenance and updates
- Better community support and documentation

## What's Preserved Here?

### Robot Node Code (`nodes/`)
Complete hardware-level robot control implementation with:
- Modular component architecture (camera, vision, movement, calibration)
- Dual-pipeline camera system (recording + scanning)
- MADSci-compliant REST API
- Comprehensive error handling and resource management

See [nodes/README.md](nodes/README.md) for detailed technical documentation.

### Diagnostic Tools (`archived-tools/`)
Hardware diagnostic and calibration tools including:
- Camera calibration tools (checkerboard-based)
- HSV color range calibration
- Object detection testing and validation
- Performance testing (10Hz recording validation)
- Hardware diagnostic scripts

See [archived-tools/README.md](archived-tools/README.md) for complete tool documentation.

### Example Workflows (`archived-workflows/`)
Working workflow examples demonstrating:
- Basic robot movement and positioning
- Synchronized data recording (10Hz)
- Single image capture
- Transfer operations

See [archived-workflows/README.md](archived-workflows/README.md) for workflow details.

## When to Use This Implementation

### Use the Archived Implementation For:
- **Camera calibration** - Tools are hardware-specific and still applicable
- **Hardware diagnostics** - Direct hardware testing and troubleshooting
- **HSV color tuning** - Interactive calibration tools for vision system
- **Reference** - Understanding hardware-level control patterns

### Use the ROS Implementation For:
- **Active development** - All new features and improvements
- **Production workflows** - Current operational system
- **Motion planning** - MoveIT integration for complex movements
- **Standard operations** - Day-to-day robot control

## Key Differences from ROS Implementation

| Feature | Archived (Hardware-Level) | Current (ROS-Based) |
|---------|---------------------------|---------------------|
| **Control** | Direct I2C servo control | ROS + MoveIT motion planning |
| **Camera** | Direct pyorbbecsdk integration | ROS camera topics |
| **Motion Planning** | Custom trajectory generation | MoveIT planner |
| **Development** | Python + hardware libraries | ROS + Python |
| **Maintenance** | Manual hardware management | ROS ecosystem support |

## Migration Notes

If you need to reference this implementation:
1. **Camera tools** remain relevant - calibration process is the same
2. **Vision algorithms** can be adapted to ROS implementation
3. **Component patterns** demonstrate good modular architecture
4. **Diagnostic tools** useful for hardware troubleshooting

## Related Documentation

- **[Current ROS Implementation](../dofbot-pro-ros/README.md)** - Active development system
- **[Diagnostic Tools](archived-tools/README.md)** - Camera calibration and hardware testing
- **[Example Workflows](archived-workflows/README.md)** - Working workflow examples
- **[Node Architecture](nodes/README.md)** - Detailed technical documentation

## Support

For issues with:
- **Current system**: See `../dofbot-pro-ros/README.md`
- **Camera calibration**: Use tools in `archived-tools/`
- **Hardware diagnostics**: See `archived-tools/README.md`
- **Historical reference**: Review code and documentation in this directory
