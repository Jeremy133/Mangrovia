package com.example.mangroveappfinal;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.text.SimpleDateFormat;
import java.util.Date;

public class MonthlySummary extends AppCompatActivity {
    DatabaseReference reference = FirebaseDatabase.getInstance().getReference().child("Users");

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_monthly_summary);
    }
    public void showBakhaw(View v) {
        setMangroove("bakhaw");
    }

    public void showBungalon(View v){
        setMangroove("bungalon");
    }

    public void showPagatpat(View v){
        setMangroove("pagatpat");
    }

    Date date = new Date();
    String year = new SimpleDateFormat("yyyy").format(date);
    private void setMangroove(String type){
        EditText yearText = (EditText) findViewById(R.id.year);
        if(yearText.getText().toString().matches("")){
            Toast.makeText(getApplicationContext(), "Please Enter Year", Toast.LENGTH_SHORT).show();
            return;
        }else{
            year = yearText.getText().toString();
        }
        ValueEventListener mangrooveDataListener = new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                int janCount = 0;
                int febCount = 0;
                int marCount = 0;
                int aprCount = 0;
                int mayCount = 0;
                int junCount = 0;
                int julCount = 0;
                int augCount = 0;
                int sepCount = 0;
                int octCount = 0;
                int novCount = 0;
                int decCount = 0;
                for(DataSnapshot ds : dataSnapshot.getChildren()) {
                    if(ds.child(type).getValue() != null){
                        if(
                                ds.child("month").getValue().toString().contains("Jan") &&
                                        ds.child("year").getValue().toString().contains(year)
                        ){
                            janCount += Integer.parseInt(ds.child(type).getValue().toString());
                        }
                        if(
                                ds.child("month").getValue().toString().contains("Feb") &&
                                        ds.child("year").getValue().toString().contains(year)
                        ){
                            febCount += Integer.parseInt(ds.child(type).getValue().toString());
                        }
                        if(
                                ds.child("month").getValue().toString().contains("Mar") &&
                                        ds.child("year").getValue().toString().contains(year)
                        ){
                            marCount += Integer.parseInt(ds.child(type).getValue().toString());
                        }
                        if(
                                ds.child("month").getValue().toString().contains("Apr") &&
                                        ds.child("year").getValue().toString().contains(year)
                        ){
                            aprCount += Integer.parseInt(ds.child(type).getValue().toString());
                        }
                        if(
                                ds.child("month").getValue().toString().contains("May") &&
                                        ds.child("year").getValue().toString().contains(year)
                        ){
                            mayCount += Integer.parseInt(ds.child(type).getValue().toString());
                        }
                        if(
                                ds.child("month").getValue().toString().contains("Jun") &&
                                        ds.child("year").getValue().toString().contains(year)
                        ){
                            junCount += Integer.parseInt(ds.child(type).getValue().toString());
                        }
                        if(
                                ds.child("month").getValue().toString().contains("Jul") &&
                                        ds.child("year").getValue().toString().contains(year)
                        ){
                            julCount += Integer.parseInt(ds.child(type).getValue().toString());
                        }
                        if(
                                ds.child("month").getValue().toString().contains("Aug") &&
                                        ds.child("year").getValue().toString().contains(year)
                        ){
                            augCount += Integer.parseInt(ds.child(type).getValue().toString());
                        }
                        if(
                                ds.child("month").getValue().toString().contains("Sept") &&
                                        ds.child("year").getValue().toString().contains(year)
                        ){
                            sepCount += Integer.parseInt(ds.child(type).getValue().toString());
                        }
                        if(
                                ds.child("month").getValue().toString().contains("Oct") &&
                                        ds.child("year").getValue().toString().contains(year)
                        ){
                            octCount += Integer.parseInt(ds.child(type).getValue().toString());
                        }
                        if(
                                ds.child("month").getValue().toString().contains("Nov") &&
                                        ds.child("year").getValue().toString().contains(year)
                        ){
                            novCount += Integer.parseInt(ds.child(type).getValue().toString());
                        }
                        if(
                                ds.child("month").getValue().toString().contains("Dec") &&
                                        ds.child("year").getValue().toString().contains(year)
                        ){
                            decCount += Integer.parseInt(ds.child(type).getValue().toString());
                        }
                        else {
                            Toast.makeText(getApplicationContext(), "No Record Found", Toast.LENGTH_SHORT).show();
                        }
                    }
                }
                TextView  jan = (TextView) findViewById(R.id.january1);
                TextView  feb = (TextView) findViewById(R.id.january2);
                TextView  mar = (TextView) findViewById(R.id.january3);
                TextView  apr = (TextView) findViewById(R.id.january4);
                TextView  may = (TextView) findViewById(R.id.january5);
                TextView  jun = (TextView) findViewById(R.id.january6);
                TextView  jul = (TextView) findViewById(R.id.january7);
                TextView  aug = (TextView) findViewById(R.id.january8);
                TextView  sep = (TextView) findViewById(R.id.january9);
                TextView  oct = (TextView) findViewById(R.id.january10);
                TextView  nov = (TextView) findViewById(R.id.january11);
                TextView  dec = (TextView) findViewById(R.id.january12);
                jan.setText(janCount + "");
                feb.setText(febCount + "");
                mar.setText(marCount + "");
                apr.setText(aprCount + "");
                may.setText(mayCount + "");
                jun.setText(junCount + "");
                jul.setText(julCount + "");
                aug.setText(augCount + "");
                sep.setText(sepCount + "");
                oct.setText(octCount + "");
                nov.setText(novCount + "");
                dec.setText(decCount + "");
            }
            @Override
            public void onCancelled(DatabaseError databaseError) {
                System.out.println(databaseError.toException());
            }
        };
        reference.addValueEventListener(mangrooveDataListener);
    }

}