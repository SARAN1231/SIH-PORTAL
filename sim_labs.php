<?php

// RDP conf
$vmIp = 'your-vm-ip';
$rdpPort = 3389;

// Url for json response
$rdpUrl = "rdp://{$vmIp}:{$rdpPort}";

// Url as a json response (API response)
header('Content-Type: application/json');
echo json_encode(['success' => true, 'rdpUrl' => $rdpUrl]);
?>
