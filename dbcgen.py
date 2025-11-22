import pandas as pd
import math
from collections import defaultdict
import cantools
from cantools.database.can import Message, Signal
from cantools.database.conversion import BaseConversion
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TYPES_LENGTH={'BOOLEAN': 1,
            'INTEGER': 8,
            'FLOAT': 8,
            'UINT8': 8,
            'UINT16': 16,
            '': 8  # Default for unknown types
            }
if not logger.handlers: # Avoid adding duplicate handlers if already configured
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def calculate_bit_length(min_val, max_val, data_type):
    """Calculate required bit length based on data type and range"""
    if not min_val or not max_val:
        # Default sizes if range not specified
        return TYPES_LENGTH.get(data_type.upper(), 8)
    
    try:
        min_val = float(min_val)
        max_val = float(max_val)
    except (ValueError, TypeError):
        return 8  # Fallback
    
    if data_type.upper() == 'BOOLEAN':
        return 1
    elif data_type.upper() == 'FLOAT':
        return 8
    else:  # INTEGER types
        value_range = max_val - min_val
        if value_range <= 0:
            return 8
        return math.ceil(math.log2(value_range + 1))

def determine_factor(data_type):
    """Assign default factor based on data type"""
    return {
        'BOOLEAN': 1,
        'INTEGER': 1,
        'FLOAT': 0.1,
        '': 1
    }.get(data_type.upper(), 1)

def generate_dbc_data(df,types=None, ECU_NAME="ECU", MSG_ID_START=0):
    """Process Sheet3 DataFrame to create DBC-ready data"""
    # Initialize DBC data structure
    dbc_data = {
        "Message_Name": [],
        "Message_ID": [],
        "Signal_Name": [],
        "Start_Bit": [],
        "Bit_Length": [],
        "Byte_Order": [],  # 0=Intel (little), 1=Motorola (big)
        "Value_Type": [],  # 0=unsigned, 1=signed
        "Factor": [],
        "Offset": [],
        "Min": [],
        "Max": [],
        "Unit": [],
        "Receiver": [],
        "Sender": [],
        "Values": []
    }
    if types is not None:
        types = {key: value.upper() for key, value in types.items()}
        # types['type.name']=df_types['type.name'].str.upper()
        
    current_start_bit = 0
    current_end_bit=0
    # last_message = None
    msg_counter = 0
    Message_ID= MSG_ID_START  # Default ID, should be configured based on your model
    for _, row in df.iterrows():
        # Group signals by interface + exchange item (assuming this represents a message)
        # message_name = f"{row['interface.name']}_{row['exchange_item.name']}"
        message_name = f"{row['interface']}_{msg_counter}"
        
        # # Reset start bit for new message
        # if message_name != last_message:
        #     current_start_bit = 0
        #     last_message = message_name
        
        # Calculate signal properties
        data_type = str(row['type']).upper()  # or row['element.type.name']
        # if df_types:
        #     type_fild=df_types['type.name']==data_type
        
        min_val = row['min'] if pd.notna(row['min']) else None
        max_val = row['max'] if pd.notna(row['max']) else None
        
        
        bit_length = calculate_bit_length(min_val, max_val, data_type)
        factor = determine_factor(data_type)
        # Increment start bit for next signal
           # Handle overflow to next byte
        current_end_bit =current_start_bit+ bit_length  # +1 for padding to next byte

        if current_end_bit >= 64:  # CAN  supports up to 64 bytes
            msg_counter+=1
            Message_ID += 1
            current_start_bit = 0
            current_end_bit=0
            message_name = f"{row['interface']}_{msg_counter}"
            last_message = message_name
            
            
            
              
        # Add to DBC data
        dbc_data["Message_Name"].append(message_name)
        dbc_data["Message_ID"].append(Message_ID)  # Default - should be configured
        dbc_data["Signal_Name"].append(row['signal'])
        dbc_data["Start_Bit"].append(current_start_bit)
        dbc_data["Bit_Length"].append(bit_length)
        dbc_data["Byte_Order"].append(0)  # Motorola (big endian)
        # dbc_data["Value_Type"].append(0 if data_type.upper() in ['BOOLEAN', 'INTEGER'] else 1)
        dbc_data["Value_Type"].append(0)
        dbc_data["Factor"].append(factor)
        dbc_data["Offset"].append(0)
        dbc_data["Min"].append(min_val if min_val else 0)
        dbc_data["Max"].append(max_val if max_val else (2**bit_length - 1))
        dbc_data["Unit"].append(row['Unit'] or '')
        dbc_data["Sender"].append(ECU_NAME)  # Default sender
        dbc_data["Receiver"].append(ECU_NAME)  # Default receiver
        dbc_data["Values"].append(None)  # Default receiver
        
   
        current_start_bit += bit_length  # +1 for padding to next byte
        
 
    
    return pd.DataFrame(dbc_data)

def generate_dbc_from_df(df, output_dbc_file="output.dbc"):
    # Read the CSV file
    # df = pd.read_csv(df_file)
    
    # Create a new CAN database
    db = cantools.database.can.Database()
    
    # Get unique sender - receivers (nodes)
    nodes = list(df['Receiver'].unique())
    for node in nodes:
        db.nodes.append(cantools.database.can.Node(name=node))
    
    # Group signals by message
    messages = df.groupby(['Message_Name', 'Message_ID'])
    
    # Process each message
    for (msg_name, msg_id), signals in messages:
        # Create signal list for the message
        signal_list = []
        for _, signal in signals.iterrows():
            # Create signal object
            conversion = BaseConversion.factory(scale=signal['Factor'], offset=signal['Offset'])
            sig = Signal(
                name=signal['Signal_Name'],
                start=signal['Start_Bit'],
                length=signal['Bit_Length'],
                byte_order='big_endian' if signal['Byte_Order'] == 1 else 'little_endian',
                is_signed=signal['Value_Type'] == 1,
                conversion=conversion,
                # scale=signal['Factor'],
                # offset=signal['Offset'],
                minimum=signal['Min'] if pd.notna(signal['Min']) else None,
                maximum=signal['Max'] if pd.notna(signal['Max']) else None,
                unit=signal['Unit'] if pd.notna(signal['Unit']) else None,
                receivers=[signal['Receiver']]
            )
            signal_list.append(sig)
        
        # Calculate message length (in bytes) based on maximum bit position
        max_bit = max(s['Start_Bit'] + s['Bit_Length'] for _, s in signals.iterrows())
        length = (max_bit + 7) // 8  # Round up to nearest byte
        # length=8
        # Create message object
        msg = Message(
            frame_id=int(msg_id),
            name=msg_name,
            length=length,
            signals=signal_list,
            senders=[signals.iloc[0]['Receiver']],  # Assuming receiver is also sender
            cycle_time=100,
            # strict=False
        )
        
        # Add message to database
        db.messages.append(msg)
    
    # Save the database to a DBC file
    # with open(output_dbc_file, 'wb') as f:
    #     cantools.database.dump_file(db, f, 'dbc')
    cantools.database.dump_file(db, output_dbc_file, 'dbc')
    print(f"DBC file created: {output_dbc_file}")


def generate_dbc_from_df1(df_dbc, output_file="output.dbc"):
    """Create a DBC file from the prepared DataFrame"""
    logger.info("my_function called.")
    # Initialize the database
    db = cantools.database.Database()
    db.version = "1.0"
    
    # Create nodes (ECUs)
    nodes = set(df_dbc["Receiver"])
    for node in nodes:
        db.nodes.append(cantools.database.Node(name=node))
    
    # Group signals by message
    messages = defaultdict(list)
    for _, row in df_dbc.iterrows():
        messages[(row["Message_Name"], row["Message_ID"])].append(row)
    
    # Create messages and signals
    for (msg_name, msg_id), signals in messages.items():
        # Calculate message length (round up to nearest byte)
        max_bit = max(sig["Start_Bit"] + sig["Bit_Length"] for sig in signals)
        msg_length = math.ceil(max_bit / 8)
        
        # Create signals
        cantools_signals = []
        for sig in signals:
            cantools_signals.append(
                cantools.database.Signal(
                    name=sig["Signal_Name"],
                    start=sig["Start_Bit"],
                    length=sig["Bit_Length"],
                    byte_order=('little' if sig["Byte_Order"] == 0 else 'big'),
                    is_signed=(sig["Value_Type"] == 1),
                    initial=None,
                    scale=sig["Factor"],
                    offset=sig["Offset"],
                    minimum=sig["Min"],
                    maximum=sig["Max"],
                    unit=sig["Unit"],
                    receivers=[sig["Receiver"]]
                )
            )
        
        # Create message
        message = cantools.database.Message(
            frame_id=int(msg_id, 16) if isinstance(msg_id, str) else msg_id,
            name=msg_name,
            length=msg_length,
            signals=cantools_signals,
            senders=None,
            cycle_time=None,
            comment=None
        )
        
        db.messages.append(message)
    
    # Write DBC file
    cantools.database.dump_file(db, output_file)
    print(f"Successfully created DBC file: {output_file}")


def create_dbc_from_xls1(df_file, output_dbc_file):
    # Read the CSV file
    # df = pd.read_csv(df_file)
    df=pd.read_excel(df_file, sheet_name='DBC_Export')  # Adjust the sheet name as needed
    
    # Create a new CAN database
    db = cantools.database.can.Database()
    
    # Get unique receivers (nodes)
    nodes = list(df['Receiver'].unique())
    for node in nodes:
        db.nodes.append(cantools.database.can.Node(name=node))
    
    # Group signals by message
    messages = df.groupby(['Message_Name', 'Message_ID'])
    
    # Process each message
    for (msg_name, msg_id), signals in messages:
        # Create signal list for the message
        signal_list = []
        for _, signal in signals.iterrows():
            # Create signal object
            conversion = BaseConversion.factory(scale=signal['Factor'], offset=signal['Offset'])
            sig = Signal(
                name=signal['Signal_Name'],
                start=signal['Start_Bit'],
                length=signal['Bit_Length'],
                byte_order='big_endian' if signal['Byte_Order'] == 1 else 'little_endian',
                is_signed=signal['Value_Type'] == 1,
                conversion=conversion,
                # scale=signal['Factor'],
                # offset=signal['Offset'],
                minimum=signal['Min'],
                maximum=signal['Max'],
                unit=signal['Unit'] if pd.notna(signal['Unit']) else None,
                receivers=[signal['Receiver']]
            )
            signal_list.append(sig)
        
        # Calculate message length (in bytes) based on maximum bit position
        max_bit = max(s['Start_Bit'] + s['Bit_Length'] for _, s in signals.iterrows())
        length = (max_bit + 7) // 8  # Round up to nearest byte
        # length=8
        # Create message object
        msg = Message(
            frame_id=int(msg_id),
            name=msg_name,
            length=length,
            signals=signal_list,
            senders=[signals.iloc[0]['Receiver']],  # Assuming receiver is also sender
            cycle_time=100,
            # strict=False
        )
        
        # Add message to database
        db.messages.append(msg)
    
    # Save the database to a DBC file
    # with open(output_dbc_file, 'wb') as f:
    #     cantools.database.dump_file(db, f, 'dbc')
    cantools.database.dump_file(db, output_dbc_file, 'dbc')
    print(f"DBC file created: {output_dbc_file}")
    
    
def generate_dbc_data1(df_sheet3, ECU_NAME="TCU", MSG_ID_START=0x100):
    """Process Sheet3 DataFrame to create DBC-ready data"""
    # Initialize DBC data structure
    dbc_data = {
        "Message_Name": [],
        "Message_ID": [],
        "Signal_Name": [],
        "Start_Bit": [],
        "Bit_Length": [],
        "Byte_Order": [],  # 0=Intel (little), 1=Motorola (big)
        "Value_Type": [],  # 0=unsigned, 1=signed
        "Factor": [],
        "Offset": [],
        "Min": [],
        "Max": [],
        "Unit": [],
        "Receiver": []
    }
    current_start_bit = 0
    current_end_bit=0
    # last_message = None
    msg_counter = 0
    Message_ID= MSG_ID_START  # Default ID, should be configured based on your model
    for _, row in df_sheet3.iterrows():
        # Group signals by interface + exchange item (assuming this represents a message)
        # message_name = f"{row['interface.name']}_{row['exchange_item.name']}"
        message_name = f"{row['interface.name']}_{msg_counter}"
        
        # # Reset start bit for new message
        # if message_name != last_message:
        #     current_start_bit = 0
        #     last_message = message_name
        
        # Calculate signal properties
        data_type = row['prop.type.kind.name'] #or row['element.type.name']
        min_val = row['prop.min'] 
        max_val = row['prop.max'] 
        
        bit_length = calculate_bit_length(min_val, max_val, data_type)
        factor = determine_factor(data_type)
        # Increment start bit for next signal
           # Handle overflow to next byte
        current_end_bit =current_start_bit+ bit_length  # +1 for padding to next byte

        if current_end_bit >= 64:  # CAN  supports up to 64 bytes
            msg_counter+=1
            Message_ID += 1
            current_start_bit = 0
            current_end_bit=0
            message_name = f"{row['interface.name']}_{msg_counter}"
            last_message = message_name
            
            
            
              
        # Add to DBC data
        dbc_data["Message_Name"].append(message_name)
        dbc_data["Message_ID"].append(Message_ID)  # Default - should be configured
        dbc_data["Signal_Name"].append(row['signal.name'])
        dbc_data["Start_Bit"].append(current_start_bit)
        dbc_data["Bit_Length"].append(bit_length)
        dbc_data["Byte_Order"].append(0)  # Motorola (big endian)
        # dbc_data["Value_Type"].append(0 if data_type.upper() in ['BOOLEAN', 'INTEGER'] else 1)
        dbc_data["Value_Type"].append(0)
        dbc_data["Factor"].append(factor)
        dbc_data["Offset"].append(0)
        dbc_data["Min"].append(min_val if min_val else 0)
        dbc_data["Max"].append(max_val if max_val else (2**bit_length - 1))
        dbc_data["Unit"].append(row['Unit'] or '')
        dbc_data["Receiver"].append(ECU_NAME)  # Default receiver
        
   
        current_start_bit += bit_length  # +1 for padding to next byte
        
 
    
    return pd.DataFrame(dbc_data)
