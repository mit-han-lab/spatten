module DRAMSimDPIDriver #(
    addressWidth = 32,
    dataWidth = 256,
    seed = 0
) (
    input  wire clk,
    input  wire rst,

    input  wire dram_req_valid,
    output wire dram_req_ready,
    input  wire [addressWidth-1:0] dram_req_payload_addr,
    input  wire [dataWidth-1:0]    dram_req_payload_data,
    input  wire dram_req_payload_is_write,

    output wire dram_resp_valid,
    input  wire dram_resp_ready,
    output wire [addressWidth-1:0] dram_resp_payload_addr,
    output wire [dataWidth-1:0]    dram_resp_payload_data,
    output wire dram_resp_payload_is_write
);

    // return 0 if succeed
    import "DPI-C" function int  DRAMSimDPIDriverCreate(input int dataWidth, input int seed, output longint obj);
    import "DPI-C" function void DRAMSimDPIDriverDestroy(input longint obj);
    import "DPI-C" function int  DRAMSimDPIDriverPushRequest(input longint obj, input  longint unsigned addr, input  bit [dataWidth-1:0] data, input bit is_write);
    import "DPI-C" function int  DRAMSimDPIDriverPopResponse(input longint obj, output longint unsigned addr, output bit [dataWidth-1:0] data, output bit is_write);
    import "DPI-C" function int  DRAMSimDPIDriverTick(input longint obj);

    longint obj;

    reg                    pending_req_valid;
    longint unsigned       pending_req_payload_addr;
    reg [dataWidth-1:0]    pending_req_payload_data;
    reg                    pending_req_payload_is_write;

    assign dram_req_ready = !pending_req_valid;

    reg                    pending_resp_valid;
    longint unsigned       pending_resp_payload_addr;
    reg [dataWidth-1:0]    pending_resp_payload_data;
    reg                    pending_resp_payload_is_write;

    assign dram_resp_valid = pending_resp_valid;
    assign dram_resp_payload_addr = pending_resp_payload_addr[addressWidth-1:0];
    assign dram_resp_payload_data = pending_resp_payload_data;
    assign dram_resp_payload_is_write = pending_resp_payload_is_write;

    always_ff @(posedge clk) begin
        if (rst) begin
            pending_req_valid <= 0;
            pending_resp_valid <= 0;
        end else begin
            // Send Requests (s2m pipe)
            if (pending_req_valid) begin
                if (DRAMSimDPIDriverPushRequest(obj, pending_req_payload_addr, pending_req_payload_data, pending_req_payload_is_write) == 0) begin
                    pending_req_valid <= 0;
                end
            end else if (dram_req_valid) begin
                if (DRAMSimDPIDriverPushRequest(obj, dram_req_payload_addr, dram_req_payload_data, dram_req_payload_is_write) != 0) begin
                    pending_req_valid <= 1;
                    pending_req_payload_addr <= dram_req_payload_addr;
                    pending_req_payload_data <= dram_req_payload_data;
                    pending_req_payload_is_write <= dram_req_payload_is_write;
                end
            end

            // Get Response (m2s pipe)
            if (!dram_resp_valid || dram_resp_ready) begin
                if (DRAMSimDPIDriverPopResponse(obj, pending_resp_payload_addr, pending_resp_payload_data, pending_resp_payload_is_write) == 0) begin
                    pending_resp_valid <= 1;
                end else begin
                    pending_resp_valid <= 0;
                end
            end

            if (DRAMSimDPIDriverTick(obj) != 0) begin
                $display("Failed to tick DRAMSimDPIDriver.");
                $finish;
            end
        end
    end

    initial begin
        if (addressWidth > 64) begin
            $display("Address width must be less than 64");
            $finish;
        end
        if (DRAMSimDPIDriverCreate(dataWidth, seed, obj) != 0) begin
            $display("Failed to create DRAMSimDPIDriver.");
            $finish;
        end
    end

    final begin
        DRAMSimDPIDriverDestroy(obj);
    end

endmodule